import json
import time
import re
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://nitte.edu.in"

START_URL = "https://nitte.edu.in/nmamit/"

ROBOTS_URL = "https://nitte.edu.in/robots.txt"

OUTPUT_FILE = "nmamit_data.json"

USER_AGENT = "NMAMIT-GraphRAG-Crawler/1.0"

REQUEST_DELAY = 0.5

MAX_PAGES = 500


HEADERS = {
    "User-Agent": USER_AGENT
}


# ============================================================
# FILE TYPES TO SKIP
# ============================================================

SKIP_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".zip",
    ".rar",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov"
}


# ============================================================
# LOAD ROBOTS.TXT
# ============================================================

def load_robots():

    print("[1] Loading robots.txt...")

    robots = RobotFileParser()

    robots.set_url(ROBOTS_URL)

    robots.read()

    print("[OK] robots.txt loaded")

    return robots


# ============================================================
# CHECK URL
# ============================================================

def is_nmamit_url(url):

    parsed = urlparse(url)

    # Accept both versions of the domain
    if parsed.netloc not in {
        "nitte.edu.in",
        "www.nitte.edu.in"
    }:
        return False

    # IMPORTANT:
    # Only crawl inside /nmamit/
    if not parsed.path.startswith("/nmamit/"):
        return False

    return True


# ============================================================
# CHECK FILE TYPE
# ============================================================

def should_skip_url(url):

    parsed = urlparse(url)

    path = parsed.path.lower()

    for extension in SKIP_EXTENSIONS:

        if path.endswith(extension):
            return True

    return False


# ============================================================
# NORMALIZE URL
# ============================================================

def normalize_url(url):

    # Remove #section from URL
    url, _ = urldefrag(url)

    parsed = urlparse(url)

    # Force https
    scheme = "https"

    # Normalize www
    netloc = parsed.netloc.lower()

    if netloc == "www.nitte.edu.in":
        netloc = "nitte.edu.in"

    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc
    ).geturl()

    return normalized


# ============================================================
# EXTRACT TABLE
# ============================================================

def extract_table(table):

    rows = []

    for tr in table.find_all("tr"):

        cells = tr.find_all(
            ["th", "td"]
        )

        row = []

        for cell in cells:

            text = cell.get_text(
                " ",
                strip=True
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            )

            row.append(text)

        if row:
            rows.append(row)

    if not rows:
        return None

    return {
        "headers": rows[0],
        "rows": rows[1:]
    }


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT PAGE
# ============================================================

def extract_page(url, session):

    print(f"      Downloading: {url}")

    try:

        response = session.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "text/html" not in content_type:

            print("      [SKIP] Not HTML")

            return None, []


        soup = BeautifulSoup(
            response.text,
            "lxml"
        )


        # ====================================================
        # FIND INTERNAL LINKS BEFORE REMOVING ELEMENTS
        # ====================================================

        discovered_links = set()

        for a in soup.find_all(
            "a",
            href=True
        ):

            href = a["href"].strip()

            absolute_url = urljoin(
                url,
                href
            )

            absolute_url = normalize_url(
                absolute_url
            )

            if not is_nmamit_url(
                absolute_url
            ):
                continue

            if should_skip_url(
                absolute_url
            ):
                continue

            discovered_links.add(
                absolute_url
            )


        # ====================================================
        # REMOVE NOISE
        # ====================================================

        remove_tags = [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg"
        ]

        for tag_name in remove_tags:

            for tag in soup.find_all(
                tag_name
            ):

                tag.decompose()


        noise_selectors = [
            "nav",
            "footer",
            ".navbar",
            ".navigation",
            ".menu",
            ".sidebar",
            ".breadcrumb",
            ".breadcrumbs",
            ".social",
            ".share",
            ".cookie",
            ".popup",
            ".modal"
        ]

        for selector in noise_selectors:

            for element in soup.select(
                selector
            ):

                element.decompose()


        # ====================================================
        # TITLE
        # ====================================================

        title = ""

        if soup.title:

            title = clean_text(
                soup.title.get_text(
                    strip=True
                )
            )


        # ====================================================
        # HEADINGS
        # ====================================================

        headings = []

        for heading in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6"
            ]
        ):

            text = clean_text(
                heading.get_text(
                    " ",
                    strip=True
                )
            )

            if text:

                headings.append({
                    "level": heading.name,
                    "text": text
                })


        # ====================================================
        # PARAGRAPHS
        # ====================================================

        paragraphs = []

        for p in soup.find_all("p"):

            text = clean_text(
                p.get_text(
                    " ",
                    strip=True
                )
            )

            if len(text) > 20:

                paragraphs.append(text)


        # ====================================================
        # TABLES
        # ====================================================

        tables = []

        for table in soup.find_all("table"):

            extracted = extract_table(
                table
            )

            if extracted:

                tables.append(
                    extracted
                )


        # ====================================================
        # LIST ITEMS
        # ====================================================

        list_items = []

        for li in soup.find_all("li"):

            text = clean_text(
                li.get_text(
                    " ",
                    strip=True
                )
            )

            if len(text) > 15:

                list_items.append(text)


        # ====================================================
        # NEW: PRESERVE DOCUMENT ORDER
        # ====================================================

        content_blocks = []

        # Find the main content area if possible
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("body")
        )

        if main_content:

            for element in main_content.find_all(
                [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "p",
                    "table",
                    "ul",
                    "ol"
                ],
                recursive=True
            ):

                # --------------------------------------------
                # HEADINGS
                # --------------------------------------------

                if element.name in [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6"
                ]:

                    text = clean_text(
                        element.get_text(
                            " ",
                            strip=True
                        )
                    )

                    if text:

                        content_blocks.append({
                            "type": "heading",
                            "level": element.name,
                            "text": text
                        })


                # --------------------------------------------
                # PARAGRAPHS
                # --------------------------------------------

                elif element.name == "p":

                    text = clean_text(
                        element.get_text(
                            " ",
                            strip=True
                        )
                    )

                    if len(text) > 20:

                        content_blocks.append({
                            "type": "paragraph",
                            "text": text
                        })


                # --------------------------------------------
                # TABLES
                # --------------------------------------------

                elif element.name == "table":

                    extracted = extract_table(
                        element
                    )

                    if extracted:

                        content_blocks.append({
                            "type": "table",
                            "data": extracted
                        })


                # --------------------------------------------
                # LISTS
                # --------------------------------------------

                elif element.name in ["ul", "ol"]:

                    items = []

                    for li in element.find_all(
                        "li",
                        recursive=False
                    ):

                        text = clean_text(
                            li.get_text(
                                " ",
                                strip=True
                            )
                        )

                        if len(text) > 15:

                            items.append(text)

                    if items:

                        content_blocks.append({
                            "type": "list",
                            "items": items
                        })


        # ====================================================
        # CREATE PAGE OBJECT
        # ====================================================

        page_data = {

            "url": url,

            "title": title,

            # Existing data — kept
            "headings": headings,

            "paragraphs": paragraphs,

            "list_items": list_items,

            "tables": tables,

            # NEW STRUCTURED DATA
            "content_blocks": content_blocks
        }


        # ====================================================
        # CHECK CONTENT
        # ====================================================

        has_content = (
            len(paragraphs) > 0
            or len(headings) > 0
            or len(tables) > 0
            or len(list_items) > 0
        )

        if not has_content:

            print(
                "      [SKIP] No useful content"
            )

            return None, discovered_links


        print(
            f"      [OK] "
            f"{len(paragraphs)} paragraphs | "
            f"{len(headings)} headings | "
            f"{len(tables)} tables | "
            f"{len(content_blocks)} blocks | "
            f"{len(discovered_links)} links"
        )

        return page_data, discovered_links


    except Exception as error:

        print(
            f"      [ERROR] {error}"
        )

        return None, set()


# ============================================================
# MAIN CRAWLER
# ============================================================

def main():

    print()
    print("=" * 60)
    print("NMAMIT GraphRAG CRAWLER")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # ROBOTS
    # --------------------------------------------------------

    robots = load_robots()


    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session = requests.Session()

    session.headers.update(
        HEADERS
    )


    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    queue = deque()

    start_url = normalize_url(
        START_URL
    )

    queue.append(
        start_url
    )


    visited = set()

    results = []


    print()
    print("[2] Starting crawl...")
    print()


    # --------------------------------------------------------
    # CRAWL
    # --------------------------------------------------------

    while queue and len(visited) < MAX_PAGES:

        url = queue.popleft()


        # Already visited
        if url in visited:
            continue


        # Make absolutely sure URL is allowed
        if not is_nmamit_url(url):
            continue


        # Skip unwanted files
        if should_skip_url(url):
            continue


        # Check robots.txt
        if not robots.can_fetch(
            USER_AGENT,
            url
        ):

            print(
                f"[ROBOTS BLOCKED] {url}"
            )

            visited.add(url)

            continue


        visited.add(url)


        print(
            f"[{len(visited)}/{MAX_PAGES}]"
        )


        # ----------------------------------------------------
        # DOWNLOAD PAGE
        # ----------------------------------------------------

        page, new_links = extract_page(
            url,
            session
        )


        # ----------------------------------------------------
        # SAVE PAGE
        # ----------------------------------------------------

        if page:

            results.append(
                page
            )


        # ----------------------------------------------------
        # ADD NEW LINKS TO QUEUE
        # ----------------------------------------------------

        for link in new_links:

            if link not in visited:

                queue.append(
                    link
                )


        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        time.sleep(
            REQUEST_DELAY
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    print()
    print("[3] Saving data...")


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"[DONE] Pages visited: {len(visited)}"
    )

    print(
        f"[DONE] Pages saved: {len(results)}"
    )

    print(
        f"[DONE] Output: {OUTPUT_FILE}"
    )

    print("=" * 60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()