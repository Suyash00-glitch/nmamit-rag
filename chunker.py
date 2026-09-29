import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = "nmamit_data.json"
OUTPUT_FILE = "chunks.json"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):

    text = str(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# DEPARTMENT DETECTION
# ---------------------------------------------------------

def detect_department(title, url):

    text = f"{title} {url}".lower()

    departments = {
        "information science": "Information Science and Engineering",
        "computer science": "Computer Science and Engineering",
        "electronics-communication": "Electronics and Communication Engineering",
        "mechanical": "Mechanical Engineering",
        "civil": "Civil Engineering",
        "electrical-electronics": "Electrical and Electronics Engineering",
        "biotechnology": "Biotechnology",
        "ai&ds": "Artificial Intelligence and Data Science",
        "ai&ml": "Artificial Intelligence and Machine Learning",
        "cybersecurity": "Cyber Security",
        "robotics": "Robotics and Artificial Intelligence",
        "vlsi": "VLSI",
        "communication": "Communication Engineering",
        "cce": "Computer and Communication Engineering",
        "mca": "MCA"
    }

    for key, department in departments.items():

        if key in text:
            return department

    return None


# ---------------------------------------------------------
# TABLE → TEXT
# ---------------------------------------------------------

def table_to_text(table):

    lines = []

    if isinstance(table, list):

        for row in table:

            if isinstance(row, list):

                row_text = " | ".join(
                    clean_text(cell)
                    for cell in row
                    if str(cell).strip()
                )

                if row_text:
                    lines.append(row_text)

            elif isinstance(row, dict):

                row_text = " | ".join(
                    f"{k}: {v}"
                    for k, v in row.items()
                    if str(v).strip()
                )

                if row_text:
                    lines.append(row_text)

    elif isinstance(table, dict):

        for key, value in table.items():

            lines.append(
                f"{key}: {value}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------
# CREATE TEXT SPLITTER
# ---------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 5,
    chunk_overlap=CHUNK_OVERLAP * 5,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        ""
    ]
)


# ---------------------------------------------------------
# PROCESS ONE PAGE
# ---------------------------------------------------------

def process_page(page):

    url = page.get("url", "")
    title = page.get("title", "")

    blocks = page.get("content_blocks", [])

    department = detect_department(title, url)

    chunks = []

    current_section = "General"
    buffer = []

    def flush_buffer():

        nonlocal buffer

        if not buffer:
            return

        text = "\n".join(buffer).strip()

        if not text:
            buffer = []
            return

        pieces = splitter.split_text(text)

        for piece in pieces:

            piece = clean_text(piece)

            if not piece:
                continue

            chunks.append({
                "content": piece,
                "content_type": "text",
                "source_url": url,
                "page_title": title,
                "section": current_section,
                "department": department
            })

        buffer = []

    # -----------------------------------------
    # PROCESS CONTENT BLOCKS
    # -----------------------------------------

    for block in blocks:

        block_type = block.get("type")

        # =====================================
        # HEADING
        # =====================================

        if block_type == "heading":

            heading = clean_text(
                block.get("text", "")
            )

            ignored_headings = {
                "ug intake",
                "duration",
                "240",
                "4 years"
            }

            if (
                heading
                and heading.lower() not in ignored_headings
                and block.get("level") in ["h2", "h3"]
            ):

                flush_buffer()

                current_section = heading

            continue

        # =====================================
        # PARAGRAPH
        # =====================================

        if block_type == "paragraph":

            text = clean_text(
                block.get("text", "")
            )

            if text:
                buffer.append(text)

            continue

        # =====================================
        # LIST
        # =====================================

        if block_type == "list":

            items = block.get("items", [])

            if items:

                list_text = "\n".join(
                    f"- {clean_text(item)}"
                    for item in items
                    if clean_text(item)
                )

                if list_text:
                    buffer.append(list_text)

            continue

        # =====================================
        # TABLE
        # =====================================

        if block_type == "table":

            flush_buffer()

            table_text = table_to_text(
                block.get("data", [])
            )

            if table_text:

                chunks.append({
                    "content": table_text,
                    "content_type": "table",
                    "source_url": url,
                    "page_title": title,
                    "section": current_section,
                    "department": department
                })

            continue

    # -----------------------------------------
    # FLUSH REMAINING TEXT
    # -----------------------------------------

    flush_buffer()

    return chunks


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

print("=" * 60)
print("NMAMIT SECTION-AWARE CHUNKER")
print("=" * 60)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    pages = json.load(f)


all_chunks = []


for page in pages:

    page_chunks = process_page(
        page
    )

    all_chunks.extend(
        page_chunks
    )


# ---------------------------------------------------------
# ADD CHUNK IDS
# ---------------------------------------------------------

for i, chunk in enumerate(
    all_chunks,
    start=1
):

    chunk["chunk_id"] = (
        f"chunk_{i}"
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print(
    f"Pages processed : {len(pages)}"
)

print(
    f"Total chunks    : {len(all_chunks)}"
)

print(
    f"Output file     : {OUTPUT_FILE}"
)

print("=" * 60)