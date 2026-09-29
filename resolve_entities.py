import json
import re
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# CONFIG
# ============================================================

INPUT_NODES = "graph_nodes_normalized.json"
INPUT_EDGES = "graph_edges_normalized.json"

OUTPUT_NODES = "graph_nodes_resolved.json"
OUTPUT_EDGES = "graph_edges_resolved.json"
OUTPUT_REPORT = "entity_resolution_report.json"


# ============================================================
# LOAD
# ============================================================

with open(INPUT_NODES, "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open(INPUT_EDGES, "r", encoding="utf-8") as f:
    edges = json.load(f)

print("=" * 70)
print("NMAMIT ENTITY RESOLUTION")
print("=" * 70)
print(f"Input nodes : {len(nodes)}")
print(f"Input edges : {len(edges)}")


# ============================================================
# NORMALIZE NAME FOR COMPARISON
# ============================================================

def normalize_name(name):
    """
    Creates a comparison-friendly version of a node name.

    This does NOT change the actual displayed name.
    It is only used to detect possible duplicates.
    """

    name = name.casefold().strip()

    # Remove common academic/title prefixes
    prefixes = [
        "dr.",
        "dr ",
        "prof.",
        "prof ",
        "mr.",
        "mr ",
        "mrs.",
        "mrs ",
        "ms.",
        "ms ",
        "sri.",
        "sri ",
        "fr.",
        "fr ",
    ]

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                changed = True
                break

    # Replace punctuation with spaces
    name = re.sub(r"[^a-z0-9&]+", " ", name)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# TOKEN NORMALIZATION
# ============================================================

def token_set(name):
    return set(normalize_name(name).split())


# ============================================================
# SIMILARITY
# ============================================================

def similarity(a, b):

    na = normalize_name(a)
    nb = normalize_name(b)

    if not na or not nb:
        return 0.0

    if na == nb:
        return 1.0

    # Exact token match
    ta = token_set(a)
    tb = token_set(b)

    if ta == tb:
        return 0.98

    # One contains the other
    if na in nb or nb in na:
        return 0.94

    # Sequence similarity
    seq_score = SequenceMatcher(None, na, nb).ratio()

    # Token overlap
    intersection = len(ta & tb)
    union = len(ta | tb)

    token_score = intersection / union if union else 0

    return max(seq_score, token_score)


# ============================================================
# IMPORTANT:
# Only automatically merge highly safe cases.
#
# We do NOT aggressively merge departments/programs/etc.
# because "Artificial Intelligence" and
# "Artificial Intelligence & Machine Learning" are NOT
# necessarily the same entity.
# ============================================================

SAFE_THRESHOLD = 0.96


# ============================================================
# GROUP CANDIDATES
# ============================================================

groups = []

used = set()

for i in range(len(nodes)):

    if i in used:
        continue

    node = nodes[i]

    group = [i]
    used.add(i)

    for j in range(i + 1, len(nodes)):

        if j in used:
            continue

        other = nodes[j]

        # Never merge different entity types
        if node.get("type") != other.get("type"):
            continue

        score = similarity(
            node.get("name", ""),
            other.get("name", "")
        )

        if score >= SAFE_THRESHOLD:

            group.append(j)
            used.add(j)

    if len(group) > 1:
        groups.append(group)


# ============================================================
# SELECT CANONICAL NODE
# ============================================================

def choose_canonical(group):

    candidates = [nodes[i] for i in group]

    # Prefer names containing full names over very short names
    candidates.sort(
        key=lambda x: (
            len(normalize_name(x.get("name", "")).split()),
            len(x.get("name", ""))
        ),
        reverse=True
    )

    return candidates[0]


# ============================================================
# CREATE NODE MAPPING
# ============================================================

node_mapping = {}

merge_report = []

for group in groups:

    canonical = choose_canonical(group)
    canonical_id = canonical["id"]

    names = []

    for index in group:

        node_id = nodes[index]["id"]

        node_mapping[node_id] = canonical_id

        names.append(nodes[index]["name"])

    merge_report.append({
        "canonical_id": canonical_id,
        "canonical_name": canonical["name"],
        "type": canonical.get("type"),
        "merged_names": names
    })


# ============================================================
# ADD IDENTITY MAPPING FOR UNMERGED NODES
# ============================================================

for node in nodes:

    if node["id"] not in node_mapping:
        node_mapping[node["id"]] = node["id"]


# ============================================================
# BUILD NEW NODE LIST
# ============================================================

canonical_nodes = {}

for node in nodes:

    canonical_id = node_mapping[node["id"]]

    if canonical_id not in canonical_nodes:

        canonical_nodes[canonical_id] = dict(node)

    else:

        existing = canonical_nodes[canonical_id]

        # Preserve useful metadata if present
        for key, value in node.items():

            if key not in existing:
                existing[key] = value

            elif not existing[key] and value:
                existing[key] = value


resolved_nodes = list(canonical_nodes.values())


# ============================================================
# REBUILD EDGES
# ============================================================

resolved_edges = []

seen_edges = set()

for edge in edges:

    source = node_mapping.get(
        edge["source"],
        edge["source"]
    )

    target = node_mapping.get(
        edge["target"],
        edge["target"]
    )

    # Remove self-loops created by merging
    if source == target:
        continue

    key = (
        source,
        target,
        edge.get("relation", "")
    )

    if key in seen_edges:
        continue

    seen_edges.add(key)

    new_edge = dict(edge)

    new_edge["source"] = source
    new_edge["target"] = target

    resolved_edges.append(new_edge)


# ============================================================
# REPORT
# ============================================================

report = {

    "input_nodes": len(nodes),

    "output_nodes": len(resolved_nodes),

    "nodes_merged": len(nodes) - len(resolved_nodes),

    "input_edges": len(edges),

    "output_edges": len(resolved_edges),

    "edges_removed": len(edges) - len(resolved_edges),

    "merge_groups": len(groups),

    "threshold": SAFE_THRESHOLD,

    "groups": merge_report
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_NODES,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resolved_nodes,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    OUTPUT_EDGES,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resolved_edges,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("ENTITY RESOLUTION COMPLETE")
print("=" * 70)

print(f"Nodes before   : {len(nodes)}")
print(f"Nodes after    : {len(resolved_nodes)}")
print(f"Nodes merged   : {len(nodes) - len(resolved_nodes)}")

print()

print(f"Edges before   : {len(edges)}")
print(f"Edges after    : {len(resolved_edges)}")
print(f"Edges removed  : {len(edges) - len(resolved_edges)}")

print()

print(f"Merge groups   : {len(groups)}")

print()
print("Output:")
print(f"  {OUTPUT_NODES}")
print(f"  {OUTPUT_EDGES}")
print(f"  {OUTPUT_REPORT}")

print("=" * 70)
