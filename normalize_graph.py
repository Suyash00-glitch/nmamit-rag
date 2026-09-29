import json
import re
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_NODES = "graph_nodes.json"
INPUT_EDGES = "graph_edges.json"

OUTPUT_NODES = "graph_nodes_normalized.json"
OUTPUT_EDGES = "graph_edges_normalized.json"
OUTPUT_REPORT = "graph_normalization_report.json"


# ============================================================
# LOAD
# ============================================================

with open(INPUT_NODES, "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open(INPUT_EDGES, "r", encoding="utf-8") as f:
    edges = json.load(f)

print("=" * 70)
print("NMAMIT GRAPH NORMALIZATION")
print("=" * 70)

print(f"Input nodes : {len(nodes)}")
print(f"Input edges : {len(edges)}")


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_text(text):
    """
    Basic normalization used only for comparing names.
    Does NOT change the displayed name.
    """

    text = text.casefold()

    # remove common punctuation
    text = re.sub(r"[.,;:()\-–—/&]", " ", text)

    # normalize common words
    replacements = {
        " and ": " ",
        " dept ": " department ",
        " deptof ": " department ",
        " engg ": " engineering ",
        " eng ": " engineering ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove department wording when comparing department/program variants
    text = re.sub(r"\bthe\b", " ", text)

    # whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def canonical_key(node):
    """
    Generates a comparison key.

    IMPORTANT:
    We do NOT merge different node types automatically.
    """

    name = node["name"]
    node_type = node["type"]

    n = normalize_text(name)

    # --------------------------------------------------------
    # INFORMATION SCIENCE
    # --------------------------------------------------------

    if node_type == "Department":

        if "information science" in n and "engineering" in n:
            return ("Department", "information science engineering")

        if "computer science" in n and "engineering" in n:
            return ("Department", "computer science engineering")

        if "artificial intelligence" in n and "machine learning" in n:
            return ("Department", "artificial intelligence machine learning")

        if "artificial intelligence" in n and "data science" in n:
            return ("Department", "artificial intelligence data science")

    # --------------------------------------------------------
    # PROGRAMS
    # --------------------------------------------------------

    if node_type == "Program":

        if "information science" in n and "engineering" in n:
            return ("Program", "information science engineering")

        if "computer science" in n and "engineering" in n:
            return ("Program", "computer science engineering")

        if "artificial intelligence" in n and "machine learning" in n:
            return ("Program", "artificial intelligence machine learning")

        if "artificial intelligence" in n and "data science" in n:
            return ("Program", "artificial intelligence data science")

        if "cyber security" in n:
            return ("Program", "cyber security")

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    if node_type == "Person":

        # Remove common title prefixes
        n = re.sub(
            r"\b(dr|prof|mr|mrs|ms|sri)\b",
            " ",
            n
        )

        n = re.sub(r"\s+", " ", n).strip()

        return ("Person", n)

    # --------------------------------------------------------
    # ORGANIZATION
    # --------------------------------------------------------

    if node_type == "Organization":

        # NMAMIT variants
        if (
            "nmam institute of technology" in n
            or n == "nmamit"
            or "nmam institute of technology nmamit" in n
        ):
            return ("Organization", "nmam institute of technology")

    # --------------------------------------------------------
    # EVERYTHING ELSE
    # --------------------------------------------------------

    return (node_type, n)


# ============================================================
# FIND DUPLICATES
# ============================================================

groups = defaultdict(list)

for node in nodes:
    key = canonical_key(node)
    groups[key].append(node)


# ============================================================
# CREATE CANONICAL NODES
# ============================================================

canonical_nodes = []
old_to_new = {}

merge_report = []

counter = 1

for key, group in groups.items():

    # --------------------------------------------------------
    # Select canonical node
    # --------------------------------------------------------

    # Prefer names without noisy prefixes/suffixes.
    group_sorted = sorted(
        group,
        key=lambda n: (
            len(n["name"]),
            n["name"]
        )
    )

    canonical = group_sorted[0]

    new_id = f"node_{counter}"
    counter += 1

    new_node = {
        "id": new_id,
        "name": canonical["name"],
        "type": canonical["type"]
    }

    canonical_nodes.append(new_node)

    # map every old node ID -> canonical ID
    for old_node in group:
        old_to_new[old_node["id"]] = new_id

    # report merges
    if len(group) > 1:

        merge_report.append({
            "canonical": {
                "id": new_id,
                "name": canonical["name"],
                "type": canonical["type"]
            },
            "merged": [
                {
                    "id": n["id"],
                    "name": n["name"],
                    "type": n["type"]
                }
                for n in group
            ]
        })


# ============================================================
# REMAP EDGES
# ============================================================

new_edges = []
seen_edges = set()

for edge in edges:

    old_source = edge["source"]
    old_target = edge["target"]

    # safety
    if old_source not in old_to_new:
        continue

    if old_target not in old_to_new:
        continue

    source = old_to_new[old_source]
    target = old_to_new[old_target]

    # remove self relationships created by merging
    if source == target:
        continue

    relation = edge["relation"]

    edge_key = (
        source,
        relation,
        target
    )

    if edge_key in seen_edges:
        continue

    seen_edges.add(edge_key)

    source_node = canonical_nodes[
        int(source.split("_")[1]) - 1
    ]

    target_node = canonical_nodes[
        int(target.split("_")[1]) - 1
    ]

    new_edges.append({
        "source": source,
        "relation": relation,
        "target": target,
        "source_name": source_node["name"],
        "target_name": target_node["name"]
    })


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_NODES, "w", encoding="utf-8") as f:
    json.dump(
        canonical_nodes,
        f,
        indent=2,
        ensure_ascii=False
    )

with open(OUTPUT_EDGES, "w", encoding="utf-8") as f:
    json.dump(
        new_edges,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# REPORT
# ============================================================

report = {
    "input_nodes": len(nodes),
    "output_nodes": len(canonical_nodes),

    "input_edges": len(edges),
    "output_edges": len(new_edges),

    "nodes_removed": len(nodes) - len(canonical_nodes),
    "edges_removed": len(edges) - len(new_edges),

    "merge_groups": len(merge_report),

    "merges": merge_report
}

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("NORMALIZATION COMPLETE")
print("=" * 70)

print(f"Nodes before : {len(nodes)}")
print(f"Nodes after  : {len(canonical_nodes)}")
print(f"Nodes merged : {len(nodes) - len(canonical_nodes)}")

print()

print(f"Edges before : {len(edges)}")
print(f"Edges after  : {len(new_edges)}")
print(f"Edges removed: {len(edges) - len(new_edges)}")

print()

print(f"Merge groups : {len(merge_report)}")

print()
print("Output:")
print(f"  {OUTPUT_NODES}")
print(f"  {OUTPUT_EDGES}")
print(f"  {OUTPUT_REPORT}")

print("=" * 70)
