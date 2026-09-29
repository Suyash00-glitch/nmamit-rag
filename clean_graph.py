import json
import re
from collections import defaultdict, Counter

INPUT = "graph_data.json"
NODES_OUTPUT = "graph_nodes.json"
EDGES_OUTPUT = "graph_edges.json"
REPORT_OUTPUT = "graph_cleaning_report.json"


# ============================================================
# TYPE NORMALIZATION
# ============================================================

TYPE_MAP = {
    "Conference": "Event",
    "Position": "JobRole",
}


# ============================================================
# RELATION NORMALIZATION
# ============================================================

RELATION_MAP = {
    "HAS_PARTICIPATED_IN": "PARTICIPATED_IN",
}


# ============================================================
# RELATION ENDPOINT TYPE INFERENCE
# ============================================================
# Used only when an endpoint is missing from the extracted
# entity list. This prevents creating unnecessary Unknown nodes.

RELATION_ENDPOINT_TYPES = {
    "HAS_ROLE": ("Person", "JobRole"),
    "WORKS_AT": ("Person", "Organization"),
    "PARTICIPATED_IN": ("Person", "Event"),
    "HAS_FACULTY": ("Course", "Person"),
    "OFFERS": ("Organization", "Program"),
    "HAS_FACILITY": ("Organization", "Facility"),
    "COLLABORATED_WITH": ("Organization", "Organization"),
    "HAS_PROGRAM": ("Organization", "Program"),
    "LOCATED_IN": (None, "Location"),
    "WORKS_ON": ("Person", "ResearchTopic"),
    "PLACED_AT": ("Placement", "Company"),
    "HAS_PLACEMENT": ("Organization", "Placement"),
    "ESTABLISHED_IN": ("Organization", "Year"),
    "HAS_RESEARCH_AREA": ("Organization", "ResearchArea"),
    "RECRUITED_BY": ("Person", "Company"),
    "OCCURS_AT": ("Event", "Location"),
    "OCCURS_IN": ("Event", "Location"),
    "HAS_ACHIEVEMENT": ("Person", "Achievement"),
    "HAS_PLACEMENT_RATE": ("PlacementStatistic", "PlacementStatistic"),
    "ORGANIZED_BY": ("Event", "Organization"),
}


# ============================================================
# LOAD RAW GRAPH
# ============================================================

with open(INPUT, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("=" * 70)
print("NMAMIT GRAPH CLEANING")
print("=" * 70)
print(f"Input chunks: {len(chunks)}")


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name):
    if not isinstance(name, str):
        return name
    return " ".join(name.strip().split())


# ============================================================
# CONSERVATIVE CANONICAL ALIASES
# ============================================================
# Explicit aliases observed during graph testing.
# No fuzzy/all-vs-all matching is used.

CANONICAL_ALIASES = {
    "nmaminstituteoftechnologynmamit": "NMAM Institute of Technology",
    "nmamit": "NMAM Institute of Technology",

    "drniranjannchiplunkar": "Dr. Niranjan N. Chiplunkar",
    "profdrniranjannchiplunkar": "Dr. Niranjan N. Chiplunkar",
    "profniranjannarayanchiplunkar": "Dr. Niranjan N. Chiplunkar",

    "informationscienceandengineering": "Information Science & Engineering",
    "informationscienceengineering": "Information Science & Engineering",
}


def matching_name(name):
    """
    Conservative matching:
    - trims whitespace
    - collapses whitespace
    - ignores case
    - ignores punctuation
    - applies only explicit canonical aliases

    No fuzzy matching is performed.
    """
    name = normalize_name(name)
    key = re.sub(r"[^a-z0-9]+", "", name.casefold())
    canonical = CANONICAL_ALIASES.get(key, key)
    return re.sub(r"[^a-z0-9]+", "", canonical.casefold())


def canonical_display_name(name):
    """Return canonical display name for an explicit alias."""
    name = normalize_name(name)
    key = re.sub(r"[^a-z0-9]+", "", name.casefold())
    return CANONICAL_ALIASES.get(key, name)


def normalize_type(entity_type):
    if not isinstance(entity_type, str):
        return entity_type

    entity_type = entity_type.strip()

    if entity_type in TYPE_MAP:
        type_changes[(entity_type, TYPE_MAP[entity_type])] += 1
        return TYPE_MAP[entity_type]

    return entity_type


def create_node(name, entity_type):
    key = (matching_name(name), entity_type)

    if key not in nodes:
        nodes[key] = {
            "id": f"node_{len(nodes) + 1}",
            "name": canonical_display_name(name),
            "type": entity_type,
        }

    return nodes[key]["id"], key


def find_candidates(name):
    """
    Find all nodes whose normalized names match.
    """
    m = matching_name(name)
    return [key for key in nodes if key[0] == m]


def find_typed_candidate(name, expected_type):
    """
    Find a matching node of the expected type.
    """
    if expected_type is None:
        return None

    m = matching_name(name)

    for key in nodes:
        if key[0] == m and key[1] == expected_type:
            return key

    return None


# ============================================================
# NODE COLLECTION
# ============================================================

nodes = {}
node_sources = defaultdict(set)

type_changes = Counter()

raw_entity_count = 0

for chunk in chunks:

    chunk_id = chunk.get("chunk_id")

    for entity in chunk.get("entities", []):

        if not isinstance(entity, dict):
            continue

        name = entity.get("name")
        entity_type = entity.get("type")

        if not name or not entity_type:
            continue

        raw_entity_count += 1

        name = normalize_name(name)
        entity_type = normalize_type(entity_type)

        node_id, key = create_node(name, entity_type)

        node_sources[key].add(chunk_id)


# ============================================================
# EDGE COLLECTION
# ============================================================

edges = {}

raw_relationship_count = 0
invalid_relationships = 0
relation_changes = Counter()

unknown_nodes = set()
unresolved_names = Counter()
inferred_endpoint_types = Counter()


def resolve_endpoint(name, relation, side):
    """
    Resolve an endpoint.

    Priority:
      1. Exact normalized typed match if relation gives an expected type.
      2. Any normalized match already present in the graph.
      3. Infer the endpoint type from the relation.
      4. Fall back to Unknown.
    """

    name = normalize_name(name)
    candidates = find_candidates(name)

    expected = None

    if relation in RELATION_ENDPOINT_TYPES:
        expected = (
            RELATION_ENDPOINT_TYPES[relation][0]
            if side == "source"
            else RELATION_ENDPOINT_TYPES[relation][1]
        )

    # --------------------------------------------------------
    # Prefer relation-inferred type
    # --------------------------------------------------------

    if expected:
        typed = find_typed_candidate(name, expected)

        if typed:
            return nodes[typed]["id"]

    # --------------------------------------------------------
    # Existing candidate
    # --------------------------------------------------------

    if candidates:
        return nodes[candidates[0]]["id"]

    # --------------------------------------------------------
    # Infer missing endpoint
    # --------------------------------------------------------

    if expected:
        node_id, key = create_node(name, expected)
        inferred_endpoint_types[expected] += 1
        return node_id

    # --------------------------------------------------------
    # Last resort: Unknown
    # --------------------------------------------------------

    node_id, key = create_node(name, "Unknown")
    unknown_nodes.add(node_id)
    unresolved_names[name] += 1

    return node_id


for chunk in chunks:

    chunk_id = chunk.get("chunk_id")

    for relationship in chunk.get("relationships", []):

        if not isinstance(relationship, dict):
            invalid_relationships += 1
            continue

        source = relationship.get("source")
        relation = relationship.get("relation")
        target = relationship.get("target")

        if not source or not relation or not target:
            invalid_relationships += 1
            continue

        raw_relationship_count += 1

        source = normalize_name(source)
        target = normalize_name(target)
        relation = relation.strip()

        if relation in RELATION_MAP:
            new_relation = RELATION_MAP[relation]
            relation_changes[(relation, new_relation)] += 1
            relation = new_relation

        source_id = resolve_endpoint(source, relation, "source")
        target_id = resolve_endpoint(target, relation, "target")

        edge_key = (
            source_id,
            relation,
            target_id,
        )

        if edge_key not in edges:
            edges[edge_key] = {
                "source": source_id,
                "relation": relation,
                "target": target_id,
                "source_name": source,
                "target_name": target,
            }


# ============================================================
# REMOVE UNKNOWN NODES THAT WERE LATER RESOLVED
# ============================================================
# Re-run endpoint matching after inference so that a name that
# became known through another relation does not remain Unknown.

unknown_keys = [
    key for key in nodes
    if key[1] == "Unknown"
]

for key in unknown_keys:

    name_match = key[0]
    candidates = [
        k for k in nodes
        if k[0] == name_match and k[1] != "Unknown"
    ]

    if not candidates:
        continue

    unknown_id = nodes[key]["id"]
    replacement_id = nodes[candidates[0]]["id"]

    for edge in edges.values():

        if edge["source"] == unknown_id:
            edge["source"] = replacement_id

        if edge["target"] == unknown_id:
            edge["target"] = replacement_id

    del nodes[key]


# ============================================================
# REBUILD EDGE DICTIONARY AFTER UNKNOWN RESOLUTION
# ============================================================

deduped_edges = {}

for edge in edges.values():

    key = (
        edge["source"],
        edge["relation"],
        edge["target"],
    )

    deduped_edges[key] = edge

edges = deduped_edges


# ============================================================
# CONVERT TO LISTS
# ============================================================

node_list = list(nodes.values())
edge_list = list(edges.values())


# ============================================================
# NODE TYPE COUNTS
# ============================================================

node_type_counts = Counter(
    node["type"]
    for node in node_list
)


# ============================================================
# EDGE TYPE COUNTS
# ============================================================

edge_type_counts = Counter(
    edge["relation"]
    for edge in edge_list
)


# ============================================================
# DUPLICATE STATISTICS
# ============================================================

duplicate_entity_mentions = raw_entity_count - len(node_list)
duplicate_relationships = raw_relationship_count - len(edge_list)


# ============================================================
# FINAL UNKNOWN / UNRESOLVED INFORMATION
# ============================================================

unknown_nodes_final = [
    node for node in node_list
    if node["type"] == "Unknown"
]

still_unresolved = Counter(
    node["name"]
    for node in unknown_nodes_final
)


# ============================================================
# SAVE NODES
# ============================================================

with open(NODES_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        node_list,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE EDGES
# ============================================================

with open(EDGES_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        edge_list,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "input": INPUT,
    "chunks": len(chunks),

    "raw_entity_mentions": raw_entity_count,
    "unique_nodes": len(node_list),
    "duplicate_entity_mentions": duplicate_entity_mentions,

    "raw_relationships": raw_relationship_count,
    "unique_edges": len(edge_list),
    "duplicate_relationships": duplicate_relationships,
    "invalid_relationships": invalid_relationships,

    "unknown_nodes": len(unknown_nodes_final),

    "unique_unresolved_names": len(still_unresolved),

    "inferred_endpoint_types": dict(
        inferred_endpoint_types
    ),

    "type_normalizations": {
        f"{old} -> {new}": count
        for (old, new), count in type_changes.items()
    },

    "relation_normalizations": {
        f"{old} -> {new}": count
        for (old, new), count in relation_changes.items()
    },

    "canonical_aliases": CANONICAL_ALIASES,

    "node_types": dict(node_type_counts),
    "edge_types": dict(edge_type_counts),

    "still_unresolved": dict(still_unresolved),
}


with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"Chunks                    : {len(chunks)}")
print(f"Raw entity mentions       : {raw_entity_count}")
print(f"Unique nodes              : {len(node_list)}")
print(f"Duplicate entity mentions: {duplicate_entity_mentions}")

print()

print(f"Raw relationships         : {raw_relationship_count}")
print(f"Unique edges              : {len(edge_list)}")
print(f"Duplicate relationships   : {duplicate_relationships}")
print(f"Invalid relationships     : {invalid_relationships}")

print()

print(f"Unknown nodes             : {len(unknown_nodes_final)}")
print(f"Unique unresolved names   : {len(still_unresolved)}")

print()

print("Inferred endpoint types:")
for entity_type, count in inferred_endpoint_types.most_common():
    print(f"  {entity_type:<25} {count}")

print()

print("Node types:")
for entity_type, count in node_type_counts.most_common():
    print(f"  {entity_type:<25} {count}")

print()

print("Relationship types:")
for relation, count in edge_type_counts.most_common():
    print(f"  {relation:<25} {count}")

print()

if still_unresolved:
    print("Still unresolved:")
    for name, count in still_unresolved.most_common():
        print(f"  {count:<5} {name}")

print()

print(f"Nodes output : {NODES_OUTPUT}")
print(f"Edges output : {EDGES_OUTPUT}")
print(f"Report       : {REPORT_OUTPUT}")

print("=" * 70)
