import json
import re
import unicodedata
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

NODES_FILE = "graph_nodes_resolved.json"
EDGES_FILE = "graph_edges_resolved.json"


# ============================================================
# LOAD GRAPH
# ============================================================

with open(NODES_FILE, "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open(EDGES_FILE, "r", encoding="utf-8") as f:
    edges = json.load(f)


# ============================================================
# BUILD LOOKUPS
# ============================================================

node_by_id = {
    node["id"]: node
    for node in nodes
}

edges_from = defaultdict(list)
edges_to = defaultdict(list)

for edge in edges:
    edges_from[edge["source"]].append(edge)
    edges_to[edge["target"]].append(edge)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text so that small formatting differences
    don't prevent matching.

    Examples:

    Dr. Niranjan N. Chiplunkar
    Niranjan.N Chiplunkar

    become approximately:

    niranjan n chiplunkar
    """

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.casefold()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokens(text):
    return set(normalize_text(text).split())


# ============================================================
# NAME SIMILARITY
# ============================================================

def similarity_score(query, name):
    """
    Simple token-based similarity.

    Gives a score between 0 and 1.
    """

    q_tokens = tokens(query)
    n_tokens = tokens(name)

    if not q_tokens or not n_tokens:
        return 0

    intersection = q_tokens & n_tokens

    # Jaccard-like similarity
    union = q_tokens | n_tokens

    return len(intersection) / len(union)


# ============================================================
# FIND NODES
# ============================================================

def find_nodes(query, limit=10):
    """
    Search nodes using multiple strategies.
    """

    query_norm = normalize_text(query)

    results = []

    for node in nodes:

        name = node.get("name", "")
        name_norm = normalize_text(name)

        score = 0

        # ----------------------------------------------------
        # Exact normalized match
        # ----------------------------------------------------

        if query_norm == name_norm:
            score = 1.0

        # ----------------------------------------------------
        # Query contained inside node name
        # ----------------------------------------------------

        elif query_norm in name_norm:
            score = 0.9

        # ----------------------------------------------------
        # Node name contained inside query
        # ----------------------------------------------------

        elif name_norm in query_norm:
            score = 0.85

        # ----------------------------------------------------
        # Token similarity
        # ----------------------------------------------------

        else:
            score = similarity_score(query, name)

        if score > 0:
            results.append(
                (
                    score,
                    node
                )
            )

    # Highest score first
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:limit]


# ============================================================
# SHOW NODE
# ============================================================

def show_node(node):

    print("\n" + "=" * 80)

    print(f"NODE: {node.get('name')}")
    print(f"TYPE: {node.get('type')}")
    print(f"ID  : {node.get('id')}")

    print("=" * 80)

    # --------------------------------------------------------
    # OUTGOING
    # --------------------------------------------------------

    print("\nOUTGOING:")

    outgoing = edges_from.get(node["id"], [])

    if not outgoing:
        print("  None")

    for edge in outgoing:

        target = node_by_id.get(edge["target"])

        if not target:
            continue

        print(
            f"  --{edge['relation']}--> "
            f"{target['name']} "
            f"[{target['type']}]"
        )

    # --------------------------------------------------------
    # INCOMING
    # --------------------------------------------------------

    print("\nINCOMING:")

    incoming = edges_to.get(node["id"], [])

    if not incoming:
        print("  None")

    for edge in incoming:

        source = node_by_id.get(edge["source"])

        if not source:
            continue

        print(
            f"  <--{edge['relation']}-- "
            f"{source['name']} "
            f"[{source['type']}]"
        )


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search(query):

    print("\n")
    print("#" * 90)
    print(f"SEARCH: {query}")
    print("#" * 90)

    matches = find_nodes(query)

    print(f"Matches: {len(matches)}")

    if not matches:
        print("  No matching nodes found.")
        return

    for score, node in matches:

        print(
            f"\nSCORE: {score:.3f}"
        )

        show_node(node)


# ============================================================
# GRAPH STATISTICS
# ============================================================

def graph_statistics():

    print("\n")
    print("=" * 80)
    print("GRAPH STATISTICS")
    print("=" * 80)

    print(f"Total nodes : {len(nodes)}")
    print(f"Total edges : {len(edges)}")

    # Count node types
    type_counts = defaultdict(int)

    for node in nodes:
        type_counts[node.get("type", "UNKNOWN")] += 1

    print("\nNODE TYPES:")

    for node_type, count in sorted(
        type_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"  {node_type:<25} {count}"
        )

    # Relation counts
    relation_counts = defaultdict(int)

    for edge in edges:
        relation_counts[edge.get("relation", "UNKNOWN")] += 1

    print("\nRELATIONS:")

    for relation, count in sorted(
        relation_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"  {relation:<30} {count}"
        )


# ============================================================
# TEST QUERIES
# ============================================================

tests = [

    # Organization
    "NMAM Institute of Technology",

    # Person with different formatting
    "Dr. Niranjan N. Chiplunkar",

    # Shorter person query
    "Niranjan Chiplunkar",

    # Department / program
    "Information Science & Engineering",

    # Department with different capitalization
    "Department of Information Science and Engineering",

    # AI
    "Artificial Intelligence",

    # NSS
    "National Service Scheme",

    # Facility
    "Research and Innovation Center",

]


# ============================================================
# RUN TESTS
# ============================================================

for query in tests:
    search(query)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("NORMALIZED GRAPH TEST COMPLETE")
print("=" * 80)

graph_statistics()

print("=" * 80)