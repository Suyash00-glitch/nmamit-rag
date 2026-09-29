
import json
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "graph_data.json"

# Delay between successful API calls
REQUEST_DELAY = 2

# Maximum retries for temporary errors such as 429
MAX_RETRIES = 5

# Initial retry delay
INITIAL_RETRY_DELAY = 10


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

total = len(chunks)

print("=" * 60)
print("NMAMIT GRAPH EXTRACTION")
print("=" * 60)
print(f"Total chunks: {total}")
print()


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print(f"Found existing {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # Build lookup of already processed chunks
    processed_ids = {
        item["chunk_id"]
        for item in graph_data
        if "chunk_id" in item
    }

    print(
        f"Already processed: "
        f"{len(processed_ids)}/{total}"
    )

else:

    graph_data = []
    processed_ids = set()

    print("Starting from beginning.")


# ============================================================
# GROQ MODEL
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2000
)

# IMPORTANT:
# Do NOT use with_structured_output().
#
# We use JSON mode because the previous tool-call approach
# caused malformed tool-call errors.

json_llm = llm.bind(
    response_format={"type": "json_object"}
)


# ============================================================
# PROMPT
# ============================================================

def create_prompt(chunk):

    return f"""
You are extracting knowledge graph information from the
official NMAM Institute of Technology (NMAMIT) website.

Extract ONLY information explicitly stated in the text.

DO NOT invent facts.

DO NOT infer relationships from:
- section names
- headings
- HTML structure
- ordering
- proximity
- assumptions

Only create a relationship when the text explicitly supports it.

Possible entity types:

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

Possible relationships:

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

IMPORTANT:

If the text is simply a list of programs, companies,
events, people, or other items and does not explicitly
state relationships between them, extract the items
as entities and return an empty relationships list.

Do NOT infer:

Department -> OFFERS -> Program

just because a department and program appear together.

If a relationship is uncertain, DO NOT create it.

Prefer fewer accurate relationships over speculative ones.

Return ONLY valid JSON.

The JSON MUST have exactly this structure:

{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type"
        }}
    ],
    "relationships": [
        {{
            "source": "source entity",
            "relation": "RELATIONSHIP",
            "target": "target entity"
        }}
    ]
}}

If there are no entities:

{{
    "entities": [],
    "relationships": []
}}

If there are no relationships:

{{
    "entities": [...],
    "relationships": []
}}

Website URL:
{chunk["source_url"]}

Page title:
{chunk["page_title"]}

Section:
{chunk.get("section")}

Department:
{chunk.get("department")}

Text:
{chunk["content"]}
"""


# ============================================================
# VALIDATE RESPONSE
# ============================================================

def validate_result(data):

    if not isinstance(data, dict):
        raise ValueError(
            "LLM response is not a JSON object"
        )

    if "entities" not in data:
        raise ValueError(
            "Missing 'entities' field"
        )

    if "relationships" not in data:
        raise ValueError(
            "Missing 'relationships' field"
        )

    if not isinstance(data["entities"], list):
        raise ValueError(
            "'entities' must be a list"
        )

    if not isinstance(data["relationships"], list):
        raise ValueError(
            "'relationships' must be a list"
        )

    # Validate entities
    for entity in data["entities"]:

        if not isinstance(entity, dict):
            raise ValueError(
                "Entity is not an object"
            )

        if "name" not in entity:
            raise ValueError(
                "Entity missing name"
            )

        if "type" not in entity:
            raise ValueError(
                "Entity missing type"
            )

    # Validate relationships
    for relationship in data["relationships"]:

        if not isinstance(relationship, dict):
            raise ValueError(
                "Relationship is not an object"
            )

        if "source" not in relationship:
            raise ValueError(
                "Relationship missing source"
            )

        if "relation" not in relationship:
            raise ValueError(
                "Relationship missing relation"
            )

        if "target" not in relationship:
            raise ValueError(
                "Relationship missing target"
            )


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(data):

    # Write to temporary file first.
    # This prevents losing the existing file if the
    # program crashes while writing.

    temp_file = OUTPUT_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temp_file,
        OUTPUT_FILE
    )


# ============================================================
# EXTRACT ONE CHUNK
# ============================================================

def extract_chunk(chunk):

    prompt = create_prompt(chunk)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"   → API call "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            response = json_llm.invoke(prompt)

            raw = response.content

            data = json.loads(raw)

            validate_result(data)

            return data

        except Exception as e:

            error_text = str(e)

            print(
                f"   → ERROR: {error_text[:300]}"
            )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if (
                "429" in error_text
                or "Too Many Requests" in error_text
                or "rate_limit" in error_text.lower()
            ):

                if attempt == MAX_RETRIES:

                    raise RuntimeError(
                        "Rate limit persisted after "
                        f"{MAX_RETRIES} attempts."
                    )

                delay = INITIAL_RETRY_DELAY * (
                    2 ** (attempt - 1)
                )

                print(
                    f"   → Rate limited. "
                    f"Waiting {delay}s..."
                )

                time.sleep(delay)

                continue

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            # Do NOT hammer the API with retries for
            # malformed JSON / bad requests.
            raise


# ============================================================
# PROCESS ALL CHUNKS
# ============================================================

successful = len(processed_ids)
errors = 0

try:

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_id = chunk["chunk_id"]

        # ----------------------------------------------------
        # SKIP ALREADY PROCESSED
        # ----------------------------------------------------

        if chunk_id in processed_ids:

            print(
                f"[{index}/{total}] "
                f"{chunk_id} → SKIPPED"
            )

            continue

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print(
            f"[{index}/{total}] "
            f"Processing {chunk_id}"
        )
        print("=" * 60)

        text = chunk.get(
            "content",
            ""
        ).strip()

        print(
            "Content:",
            text[:250].replace("\n", " ")
        )

        result = {
            "chunk_id": chunk_id,
            "source_url": chunk["source_url"],
            "page_title": chunk["page_title"],
            "section": chunk.get("section"),
            "department": chunk.get("department"),
            "entities": [],
            "relationships": []
        }

        try:

            extraction = extract_chunk(chunk)

            result["entities"] = (
                extraction["entities"]
            )

            result["relationships"] = (
                extraction["relationships"]
            )

            print(
                f"   → SUCCESS: "
                f"{len(result['entities'])} entities, "
                f"{len(result['relationships'])} relationships"
            )

            successful += 1

        except Exception as e:

            print(
                "   → FAILED:",
                str(e)
            )

            result["error"] = str(e)

            errors += 1

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        graph_data.append(result)

        save_progress(graph_data)

        processed_ids.add(chunk_id)

        print(
            f"   → Saved progress "
            f"({len(graph_data)}/{total})"
        )

        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        if index < total:

            print(
                f"   → Waiting "
                f"{REQUEST_DELAY}s..."
            )

            time.sleep(
                REQUEST_DELAY
            )


except KeyboardInterrupt:

    print()
    print("=" * 60)
    print("INTERRUPTED")
    print("=" * 60)

    save_progress(graph_data)

    print(
        f"Progress saved: "
        f"{len(graph_data)}/{total}"
    )

    print(
        "Run the same command again to resume."
    )

except Exception as e:

    print()
    print("=" * 60)
    print("STOPPED")
    print("=" * 60)

    print(
        "Reason:",
        str(e)
    )

    save_progress(graph_data)

    print(
        f"Progress saved: "
        f"{len(graph_data)}/{total}"
    )

    print(
        "Run the same command again to resume."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("GRAPH EXTRACTION STATUS")
print("=" * 60)

print(
    "Total chunks       :",
    total
)

print(
    "Processed chunks   :",
    len(graph_data)
)

print(
    "Successful         :",
    sum(
        1
        for x in graph_data
        if "error" not in x
    )
)

print(
    "Errors             :",
    sum(
        1
        for x in graph_data
        if "error" in x
    )
)

print(
    "Entities           :",
    sum(
        len(x["entities"])
        for x in graph_data
    )
)

print(
    "Relationships      :",
    sum(
        len(x["relationships"])
        for x in graph_data
    )
)

print(
    "Output             :",
    OUTPUT_FILE
)

print("=" * 60)

