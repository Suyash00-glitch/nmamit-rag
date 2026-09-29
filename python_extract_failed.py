import json
import re
import os


# ============================================================
# CONFIG
# ============================================================

CHUNKS_FILE = "chunks.json"
GRAPH_FILE = "graph_data.json"
OUTPUT_FILE = "python_extracted.json"


# ============================================================
# LOAD
# ============================================================

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)


# ============================================================
# FIND FAILED CHUNKS
# ============================================================

existing = {
    x["chunk_id"]: x
    for x in graph_data
}

failed_chunks = [
    chunk
    for chunk in chunks
    if chunk["chunk_id"] in existing
    and "error" in existing[chunk["chunk_id"]]
]


# ============================================================
# LOAD PREVIOUS PYTHON RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        python_data = json.load(f)

else:

    python_data = []


python_results = {
    x["chunk_id"]: x
    for x in python_data
}


# ============================================================
# ENTITY HELPERS
# ============================================================

def add_entity(entities, seen, name, entity_type):

    name = name.strip()

    if not name:
        return

    # Clean obvious punctuation
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ,.;:-")

    if len(name) < 2:
        return

    key = (name.lower(), entity_type)

    if key in seen:
        return

    seen.add(key)

    entities.append({
        "name": name,
        "type": entity_type
    })


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):

    entities = []
    seen = set()

    # --------------------------------------------------------
    # NMAMIT / Nitte organizations
    # --------------------------------------------------------

    org_patterns = [
        r"NMAM Institute of Technology",
        r"NMAM Institute of Technology \(NMAMIT\)",
        r"Nitte University",
        r"Nitte",
        r"National Service Scheme \(NSS\) Unit",
        r"NSS Unit",
        r"Rotary Club Nitte",
        r"Nitte Education Trust",
    ]

    for pattern in org_patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            add_entity(
                entities,
                seen,
                match.group(),
                "Organization"
            )

    # --------------------------------------------------------
    # Departments
    # --------------------------------------------------------

    department_patterns = [
        r"Department of [A-Z][A-Za-z &,-]+",
        r"School of [A-Z][A-Za-z &,-]+",
    ]

    for pattern in department_patterns:

        for match in re.finditer(pattern, text):

            name = match.group()

            # Stop at common punctuation
            name = re.split(
                r"[.;:\n]",
                name
            )[0]

            add_entity(
                entities,
                seen,
                name,
                "Department"
            )

    # --------------------------------------------------------
    # Engineering / academic programs
    # --------------------------------------------------------

    program_patterns = [
        r"Artificial Intelligence & Data Science",
        r"Artificial Intelligence & Machine Learning",
        r"Computer Science & Engineering",
        r"Computer & Communication Engineering",
        r"Information Science & Engineering",
        r"Electrical & Electronics Engineering",
        r"Electronics & Communication Engineering",
        r"Mechanical Engineering",
        r"Civil Engineering",
        r"Robotics & Artificial Intelligence",
        r"Electronics Engineering \(VLSI Design & Technology\)",
        r"VLSI Design and Embedded Systems",
        r"Computer Science & Engineering\(Cyber Security\)",
        r"Computer Science & Engineering \(Cyber Security\)",
        r"Construction Technology",
        r"Electric Vehicle Technology",
        r"Structural Engineering",
    ]

    for pattern in program_patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            add_entity(
                entities,
                seen,
                match.group(),
                "Program"
            )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    event_patterns = [
        r"Techno-Cultural Fest",
        r"National Pledge Against Drug Abuse",
        r"[A-Z][A-Za-z-]+ Fest",
        r"[A-Z][A-Za-z-]+ Festival",
        r"Inauguration of [A-Z][A-Za-z0-9 &-]+",
    ]

    for pattern in event_patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            add_entity(
                entities,
                seen,
                match.group(),
                "Event"
            )

    # --------------------------------------------------------
    # Locations
    # --------------------------------------------------------

    location_patterns = [
        r"Mangaluru, India",
        r"Mangalore, India",
        r"Mangalore",
        r"Mangaluru",
        r"Nitte",
        r"Karnataka",
        r"India",
    ]

    for pattern in location_patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            add_entity(
                entities,
                seen,
                match.group(),
                "Location"
            )

    # --------------------------------------------------------
    # Companies
    # --------------------------------------------------------

    company_patterns = [
        r"\b[A-Z][A-Za-z]+ (?:Technologies|Technology|Systems|Solutions|Industries|Limited|Ltd|Pvt\. Ltd\.)",
    ]

    for pattern in company_patterns:

        for match in re.finditer(
            pattern,
            text
        ):

            add_entity(
                entities,
                seen,
                match.group(),
                "Company"
            )

    # --------------------------------------------------------
    # Research / publication titles
    # --------------------------------------------------------

    # Detect sentences that look like paper titles.
    # Conservative: only if publication-like keywords exist.

    publication_keywords = [
        "performance evaluation",
        "analysis",
        "study",
        "design",
        "development",
        "implementation",
        "optimization",
        "characteristics",
        "investigation",
        "assessment",
    ]

    lower_text = text.lower()

    if any(
        keyword in lower_text
        for keyword in publication_keywords
    ):

        # Use the whole chunk as a research topic only
        # when it looks like a publication.
        if any(
            keyword in lower_text
            for keyword in [
                "proceedings",
                "journal",
                "vol.",
                "doi",
                "pp.",
                "conference"
            ]
        ):

            cleaned = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            add_entity(
                entities,
                seen,
                cleaned,
                "ResearchTopic"
            )

    return entities


# ============================================================
# RELATIONSHIP EXTRACTION
# ============================================================

def extract_relationships(text, entities):

    relationships = []

    lower = text.lower()

    entity_names = {
        e["name"].lower(): e["name"]
        for e in entities
    }

    def has_entity(name):

        return name.lower() in entity_names

    # --------------------------------------------------------
    # PARTICIPATED_IN
    # --------------------------------------------------------

    if (
        "participated in" in lower
        or "participated at" in lower
    ):

        event = None

        for e in entities:

            if e["type"] == "Event":
                event = e["name"]
                break

        organization = None

        for e in entities:

            if e["type"] in [
                "Organization",
                "Department"
            ]:

                if (
                    "nss" in e["name"].lower()
                    or "rotary" in e["name"].lower()
                ):
                    organization = e["name"]
                    break

        if organization and event:

            relationships.append({
                "source": organization,
                "relation": "PARTICIPATED_IN",
                "target": event
            })

    # --------------------------------------------------------
    # COLLABORATED_WITH
    # --------------------------------------------------------

    if (
        "in association with" in lower
        or "collaborated with" in lower
        or "in collaboration with" in lower
    ):

        orgs = [
            e["name"]
            for e in entities
            if e["type"] == "Organization"
        ]

        if len(orgs) >= 2:

            main_org = None

            for org in orgs:

                if (
                    "nmam institute" in org.lower()
                    or "nitte university" in org.lower()
                ):

                    main_org = org
                    break

            if main_org:

                for org in orgs:

                    if org != main_org:

                        relationships.append({
                            "source": main_org,
                            "relation": "COLLABORATED_WITH",
                            "target": org
                        })

    # --------------------------------------------------------
    # LOCATED_IN
    # --------------------------------------------------------

    if (
        "located in" in lower
        or "located at" in lower
        or "situated in" in lower
    ):

        organizations = [
            e["name"]
            for e in entities
            if e["type"] == "Organization"
        ]

        locations = [
            e["name"]
            for e in entities
            if e["type"] == "Location"
        ]

        if organizations and locations:

            relationships.append({
                "source": organizations[0],
                "relation": "LOCATED_IN",
                "target": locations[0]
            })

    # --------------------------------------------------------
    # OFFERS
    # --------------------------------------------------------

    if (
        "offers" in lower
        or "offered by" in lower
        or "offers the" in lower
    ):

        organizations = [
            e["name"]
            for e in entities
            if e["type"] == "Organization"
        ]

        programs = [
            e["name"]
            for e in entities
            if e["type"] == "Program"
        ]

        if organizations and programs:

            for program in programs:

                # Only create if "offers" appears near
                # the program name.
                pos = lower.find(
                    program.lower()
                )

                if pos != -1:

                    nearby = lower[
                        max(0, pos - 150):
                        pos + 150
                    ]

                    if "offer" in nearby:

                        relationships.append({
                            "source": organizations[0],
                            "relation": "OFFERS",
                            "target": program
                        })

    return relationships


# ============================================================
# PROCESS
# ============================================================

print("=" * 60)
print("NMAMIT PYTHON EXTRACTION — FAILED CHUNKS")
print("=" * 60)
print(
    f"Total chunks       : {len(chunks)}"
)
print(
    f"Failed chunks      : {len(failed_chunks)}"
)
print(
    f"Already processed  : {len(python_results)}"
)
print("=" * 60)


processed = 0


try:

    for chunk in failed_chunks:

        chunk_id = chunk["chunk_id"]

        # ----------------------------------------------------
        # Skip if already processed by this script
        # ----------------------------------------------------

        if chunk_id in python_results:
            continue

        processed += 1

        print()
        print(
            f"[{processed}] Processing {chunk_id}"
        )

        text = chunk.get(
            "content",
            ""
        ).strip()

        entities = extract_entities(text)

        relationships = extract_relationships(
            text,
            entities
        )

        result = {
            "chunk_id": chunk_id,
            "source_url": chunk["source_url"],
            "page_title": chunk["page_title"],
            "section": chunk.get("section"),
            "department": chunk.get("department"),
            "entities": entities,
            "relationships": relationships,
            "extraction_method": "python"
        }

        python_results[chunk_id] = result

        # ----------------------------------------------------
        # SAVE EVERY CHUNK
        # ----------------------------------------------------

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                list(python_results.values()),
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"→ {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )

        print(
            "→ Saved"
        )


except KeyboardInterrupt:

    print()
    print("CTRL+C detected.")
    print("Progress already saved.")


# ============================================================
# FINAL
# ============================================================

entity_count = sum(
    len(x["entities"])
    for x in python_results.values()
)

relationship_count = sum(
    len(x["relationships"])
    for x in python_results.values()
)

print()
print("=" * 60)
print("PYTHON EXTRACTION COMPLETE")
print("=" * 60)
print(
    f"Processed          : {len(python_results)}"
)
print(
    f"Entities           : {entity_count}"
)
print(
    f"Relationships      : {relationship_count}"
)
print(
    f"Output             : {OUTPUT_FILE}"
)
print("=" * 60)