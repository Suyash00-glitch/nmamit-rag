import json
import re
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

INPUT_FILE = "chunks.json"
GRAPH_FILE = "graph_data.json"

MODEL = "openai/gpt-oss-20b"

DELAY_SECONDS = 2


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# This is the ORIGINAL total, fixed for the whole run.
# It is what the denominator in progress printing must always use.
TOTAL_CHUNKS = len(chunks)


# ============================================================
# LOAD EXISTING GRAPH
# ============================================================

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)


# ============================================================
# INDEX RESULTS
# ============================================================

results_by_id = {
    item["chunk_id"]: item
    for item in graph_data
}


# ============================================================
# CHUNK NUMBER PARSING (for display only)
#
# chunk_id is expected like "chunk_78" -> 78
# Falls back to the chunk's position in chunks.json (1-based)
# if the id doesn't end in digits, so display never crashes.
# ============================================================

def get_chunk_number(chunk_id, fallback_index):
    match = re.search(r"(\d+)$", str(chunk_id))
    if match:
        return int(match.group(1))
    return fallback_index


# ============================================================
# FIND FAILED CHUNKS
#
# ONLY chunks with "error" (or missing entirely) are retried.
# Successful chunks are NEVER touched.
#
# This list is computed ONCE, from the state of graph_data.json
# on disk at the moment the script starts. It does not change
# during the run, so nothing already attempted in this run can
# be re-attempted in this run.
# ============================================================

failed_chunks = []

for position, chunk in enumerate(chunks, start=1):

    chunk_id = chunk["chunk_id"]

    existing = results_by_id.get(chunk_id)

    if existing is None:
        # Completely missing chunk
        failed_chunks.append((chunk, position))

    elif "error" in existing:
        # Previously failed chunk
        failed_chunks.append((chunk, position))

    else:
        # Already successful — skip forever
        continue


# ============================================================
# GROQ
# ============================================================

# This account's Groq tier caps total tokens (prompt + completion)
# at 8000 per request/minute. A single oversized chunk can exceed
# that on its own -- no amount of retrying fixes that, the request
# has to be sized to fit before it's ever sent.
GROQ_TPM_LIMIT = 8000
SAFETY_MARGIN = 400            # headroom below the hard cap
MIN_COMPLETION_TOKENS = 800    # floor: always leave room for a real answer
MAX_COMPLETION_TOKENS = 6000   # ceiling: matches prior working value
MAX_SIZE_RETRIES = 4           # bounded shrink-and-resend attempts per chunk


def estimate_tokens(text):
    # Rough heuristic (~4 chars/token). Used only to pick a sane
    # starting point -- not trusted to be exact, since gpt-oss's
    # real tokenizer counts noticeably higher for this content.
    return max(1, len(text) // 4)


def parse_token_limit_error(exc):
    # Groq's 413 body looks like:
    # "...on tokens per minute (TPM): Limit 8000, Requested 8963..."
    # Pulling the real numbers out beats guessing with a heuristic,
    # since it's exactly what the API measured for this request.
    match = re.search(
        r"Limit (\d+), Requested (\d+)", str(exc)
    )
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def is_token_limit_error(exc):
    msg = str(exc)
    return "rate_limit_exceeded" in msg and "tokens per minute" in msg


def build_llm(max_tokens):
    return ChatGroq(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        # gpt-oss-20b is a reasoning model: by default it spends an
        # unbounded amount of max_tokens on hidden reasoning before
        # writing the actual answer. If reasoning eats the whole
        # budget, the visible content comes back empty. Forcing low
        # reasoning effort keeps that overhead small so the real
        # JSON answer actually gets written.
        reasoning_effort="low"
    )


# ============================================================
# EXTRACTION
# ============================================================

def build_prompt(content):

    return f"""
You are extracting a knowledge graph from the official
NMAM Institute of Technology (NMAMIT) website.

Extract ONLY information explicitly stated in the text.

DO NOT invent information.

Return ONLY valid JSON.
DO NOT use markdown.
DO NOT use ```.

The response MUST contain exactly:

{{
  "entities": [],
  "relationships": []
}}

ENTITY FORMAT:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ]
}}

RELATIONSHIP FORMAT:

{{
  "relationships": [
    {{
      "source": "entity name",
      "relation": "RELATION",
      "target": "entity name"
    }}
  ]
}}

ALLOWED ENTITY TYPES:

Department
Person
Program
Course
ResearchArea
ResearchTopic
Facility
Organization
Company
Event
Placement
Location
Achievement
Alumni
PlacementRecord
JobRole
PlacementStatistic

ALLOWED RELATIONSHIPS:

OFFERS
HAS_FACULTY
WORKS_AT
WORKS_ON
HAS_PROGRAM
HAS_FACILITY
HAS_PLACEMENT
PLACED_AT
PARTICIPATED_IN
LOCATED_IN
ESTABLISHED_IN
HAS_PLACEMENT_RECORD
HAS_ROLE
HAS_PLACEMENT_RATE
RECRUITED_BY
COLLABORATED_WITH

RULES:

1. Extract only entities explicitly present in the text.
2. Extract only relationships explicitly supported by the text.
3. Do NOT infer relationships from headings.
4. Do NOT infer relationships from section names.
5. Do NOT infer relationships from ordering.
6. Do NOT infer relationships from proximity.
7. Do NOT assume nearby entities are related.
8. If a relationship is uncertain, do not create it.
9. Prefer fewer accurate relationships.
10. If nothing can be extracted, return empty lists.

TEXT:

{content}
"""


def extract_with_llm(chunk):

    content = chunk["content"]

    prompt = build_prompt(content)

    prompt_tokens = estimate_tokens(prompt)

    # Reserve room for at least a minimal completion. If the chunk's
    # own content is so long that prompt + minimum completion would
    # already blow the TPM ceiling, truncate the content itself --
    # a shortened extraction beats a guaranteed-failing request.
    max_prompt_tokens = (
        GROQ_TPM_LIMIT - SAFETY_MARGIN - MIN_COMPLETION_TOKENS
    )

    if prompt_tokens > max_prompt_tokens:

        template_overhead_tokens = estimate_tokens(build_prompt(""))

        allowed_content_tokens = max(
            200,
            max_prompt_tokens - template_overhead_tokens
        )

        allowed_content_chars = allowed_content_tokens * 4

        content = content[:allowed_content_chars]

        prompt = build_prompt(content)

        prompt_tokens = estimate_tokens(prompt)

    # Size the completion budget to whatever room is left, within
    # our floor/ceiling, so prompt + completion stays under the cap.
    remaining_tokens = GROQ_TPM_LIMIT - SAFETY_MARGIN - prompt_tokens

    completion_tokens = max(
        MIN_COMPLETION_TOKENS,
        min(MAX_COMPLETION_TOKENS, remaining_tokens)
    )

    llm = build_llm(completion_tokens)

    last_error = None

    for size_attempt in range(MAX_SIZE_RETRIES):

        try:
            response = llm.invoke(prompt)
            break  # success -- fall through to parsing below

        except Exception as e:

            if not is_token_limit_error(e):
                # Not a sizing problem -- let the normal per-chunk
                # error handling in the main loop deal with it.
                raise

            last_error = e

            limit, requested = parse_token_limit_error(e)

            if limit is None:
                # Couldn't parse the numbers -- nothing precise to
                # correct, so don't guess further.
                raise

            # Exact overage as measured by Groq, plus a buffer so
            # the next attempt doesn't just graze the limit again.
            overage = (requested - limit) + SAFETY_MARGIN

            if completion_tokens - overage >= MIN_COMPLETION_TOKENS:
                # First lever: shrink the completion budget by
                # exactly the measured overage.
                completion_tokens -= overage
                llm = build_llm(completion_tokens)

            else:
                # Completion budget is already at the floor --
                # the content itself has to shrink instead.
                # ~4 chars/token is only an estimate for this step,
                # but we loop again and re-measure the real overage
                # from Groq's response either way.
                chars_to_cut = overage * 4

                if len(content) <= chars_to_cut + 500:
                    # Nothing meaningful left to cut -- give up and
                    # let this chunk be recorded as failed.
                    raise

                content = content[:len(content) - chars_to_cut]
                prompt = build_prompt(content)
                completion_tokens = MIN_COMPLETION_TOKENS
                llm = build_llm(completion_tokens)

    else:
        # Loop exhausted every allowed shrink attempt without success.
        raise last_error

    raw = response.content

    if not raw:
        finish_reason = (
            response.response_metadata.get("finish_reason")
            if getattr(response, "response_metadata", None)
            else None
        )
        raise ValueError(
            f"Empty model response (finish_reason={finish_reason})"
        )

    raw = str(raw).strip()

    # Remove markdown fences
    if raw.startswith("```"):

        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    # Find JSON
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON found: " + raw[:500]
        )

    raw = raw[start:end + 1]

    # Parse
    data = json.loads(raw)

    if "entities" not in data:
        data["entities"] = []

    if "relationships" not in data:
        data["relationships"] = []

    if not isinstance(data["entities"], list):
        raise ValueError("entities is not a list")

    if not isinstance(data["relationships"], list):
        raise ValueError("relationships is not a list")

    return data


# ============================================================
# SAVE
# ============================================================

def save_progress():

    ordered = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]

        if chunk_id in results_by_id:
            ordered.append(
                results_by_id[chunk_id]
            )

    temp_file = GRAPH_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            ordered,
            f,
            indent=2,
            ensure_ascii=False
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temp_file,
        GRAPH_FILE
    )


# ============================================================
# STATUS
# ============================================================

successful_before = sum(
    1
    for x in graph_data
    if "error" not in x
)

failed_before = sum(
    1
    for x in graph_data
    if "error" in x
)


print()
print("=" * 60)
print("NMAMIT GRAPH EXTRACTION — FAILED CHUNKS")
print("=" * 60)

print(
    f"Total chunks       : {TOTAL_CHUNKS}"
)

print(
    f"Already successful : {successful_before}"
)

print(
    f"Previously failed  : {failed_before}"
)

print(
    f"To process         : {len(failed_chunks)}"
)

print(
    f"Model              : {MODEL}"
)

print(
    "Retries per chunk  : 0"
)

print("=" * 60)


# ============================================================
# PROCESS
# ============================================================

try:

    for count, (chunk, position) in enumerate(
        failed_chunks,
        start=1
    ):

        chunk_id = chunk["chunk_id"]

        # Display numerator = actual chunk number (parsed from
        # chunk_id, e.g. "chunk_78" -> 78), NOT the loop index.
        # Display denominator = TOTAL_CHUNKS (2243), fixed,
        # NOT len(failed_chunks).
        chunk_number = get_chunk_number(chunk_id, position)

        print()
        print("=" * 60)

        print(
            f"[{chunk_number}/{TOTAL_CHUNKS}] "
            f"Processing {chunk_id} "
            f"({count}/{len(failed_chunks)} in this run)"
        )

        print("=" * 60)

        result = {
            "chunk_id": chunk["chunk_id"],
            "source_url": chunk["source_url"],
            "page_title": chunk["page_title"],
            "section": chunk.get("section"),
            "department": chunk.get("department"),
            "entities": [],
            "relationships": []
        }

        try:

            print("→ Groq API call")

            extraction = extract_with_llm(chunk)

            result["entities"] = extraction["entities"]

            result["relationships"] = extraction["relationships"]

            print(
                f"→ SUCCESS: "
                f"{len(result['entities'])} entities, "
                f"{len(result['relationships'])} relationships"
            )

        except Exception as e:

            print(
                "→ FAILED:",
                str(e)
            )

            result["error"] = str(e)

        # ----------------------------------------------------
        # SAVE RESULT — immediately, after every single attempt
        # ----------------------------------------------------

        results_by_id[chunk_id] = result

        save_progress()

        print(
            f"→ Progress saved "
            f"({count}/{len(failed_chunks)} attempted this run)"
        )

        time.sleep(DELAY_SECONDS)


except KeyboardInterrupt:

    print()
    print("=" * 60)
    print("CTRL+C DETECTED")
    print("=" * 60)

    save_progress()

    print("Progress saved.")
    print("Run again to continue.")
    print("=" * 60)


# ============================================================
# FINAL STATUS
# ============================================================

with open(
    GRAPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    final_data = json.load(f)


successful = sum(
    1
    for x in final_data
    if "error" not in x
)

failed = sum(
    1
    for x in final_data
    if "error" in x
)

entities = sum(
    len(x.get("entities", []))
    for x in final_data
)

relationships = sum(
    len(x.get("relationships", []))
    for x in final_data
)


print()
print("=" * 60)
print("GRAPH EXTRACTION STATUS")
print("=" * 60)

print(
    f"Total chunks       : {TOTAL_CHUNKS}"
)

print(
    f"Successful         : {successful}"
)

print(
    f"Failed             : {failed}"
)

print(
    f"Entities           : {entities}"
)

print(
    f"Relationships       : {relationships}"
)

print(
    f"Output             : {GRAPH_FILE}"
)

print("=" * 60)