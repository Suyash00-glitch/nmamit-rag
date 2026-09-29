import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher


NODES_FILE = "graph_nodes.json"
EDGES_FILE = "graph_edges.json"
OUTPUT_FILE = "graph_audit_report.json"


# ============================================================
# LOAD GRAPH
# ============================================================

with open(NODES_FILE, "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open(EDGES_FILE, "r", encoding="utf-8") as f:
    edges = json.load(f)

node_by_id = {n["id"]: n for n in nodes}


# ============================================================
# CONFIG
# ============================================================

GENERIC_NAMES = {
    "faculty",
    "faculty members",
    "principal",
    "director",
    "staff",
    "students",
    "student",
    "department",
    "program",
    "course",
    "event",
    "organization",
    "company",
    "institute",
    "college",
    "university",
    "leading industries",
    "research organizations",
    "international universities",
}


# Relationships where certain target/source types are suspicious.
EXPECTED_TYPES = {
    "WORKS_AT": {
        "source": {"Person", "Alumni"},
        "target": {"Organization", "Company", "Department"},
    },

    "HAS_ROLE": {
        "source": {"Person", "Organization"},
        "target": {"JobRole", "Role"},
    },

    "HAS_FACULTY": {
        "source": {"Organization", "Department", "Program"},
        "target": {"Person"},
    },

    "OFFERS": {
        "source": {"Organization", "Department", "Program"},
        "target": {
            "Program",
            "Course",
            "Company",
            "Organization",
            "Event",
        },
    },

    "HAS_PROGRAM": {
        "source": {"Organization", "Department"},
        "target": {"Program"},
    },

    "HAS_FACILITY": {
        "source": {"Organization", "Department", "Event"},
        "target": {"Facility"},
    },

    "WORKS_ON": {
        "source": {"Person", "Organization", "Department"},
        "target": {"ResearchTopic", "ResearchProject", "Event"},
    },

    "COLLABORATED_WITH": {
        "source": {
            "Organization",
            "Department",
            "Program",
            "Company",
        },
        "target": {
            "Organization",
            "Department",
            "Program",
            "Company",
        },
    },

    "PARTICIPATED_IN": {
        "source": {
            "Person",
            "Organization",
            "Department",
            "Team",
            "Program",
        },
        "target": {"Event", "Program"},
    },

    "LOCATED_IN": {
        "source": {
            "Organization",
            "Department",
            "Facility",
            "Event",
            "Company",
            "Person",
        },
        "target": {"Location", "Facility", "Organization"},
    },

    "ESTABLISHED_IN": {
        "source": {
            "Organization",
            "Company",
            "Program",
        },
        "target": {"Location", "Organization", "Year"},
    },

    "PLACED_AT": {
        "source": {"Organization", "Department", "Placement"},
        "target": {"Company", "Organization"},
    },

    "HAS_PLACEMENT": {
        "source": {"Organization", "Department", "Program"},
        "target": {"Placement", "Company"},
    },

    "HAS_ACHIEVEMENT": {
        "source": {"Person", "Organization", "Program"},
        "target": {"Achievement"},
    },
}


# ============================================================
# 1. BASIC STATISTICS
# ============================================================

node_types = Counter(n["type"] for n in nodes)
edge_types = Counter(e["relation"] for e in edges)


# ============================================================
# 2. GENERIC NODES
# ============================================================

generic_nodes = []

for node in nodes:

    name = node["name"].strip().casefold()

    if name in GENERIC_NAMES:

        generic_nodes.append({
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
        })


# ============================================================
# 3. POSSIBLE DUPLICATE NAMES
#
# Conservative similarity check.
# We DO NOT automatically merge anything.
# ============================================================

def normalize_for_similarity(name):

    name = name.casefold()

    replacements = [
        "&",
        "(",
        ")",
        ",",
        ".",
        "-",
        "_",
        "'",
        '"',
    ]

    for char in replacements:
        name = name.replace(char, " ")

    return " ".join(name.split())


similar_candidates = []

# Group by type first.
nodes_by_type = defaultdict(list)

for node in nodes:
    nodes_by_type[node["type"]].append(node)


for entity_type, type_nodes in nodes_by_type.items():

    # Avoid O(n^2) on huge unrelated groups where possible.
    for i in range(len(type_nodes)):

        a = type_nodes[i]

        a_name = normalize_for_similarity(a["name"])

        if len(a_name) < 5:
            continue

        for j in range(i + 1, len(type_nodes)):

            b = type_nodes[j]

            b_name = normalize_for_similarity(b["name"])

            if len(b_name) < 5:
                continue

            # Exact normalized name
            if a_name == b_name:

                similar_candidates.append({
                    "reason": "same_normalized_name",
                    "type": entity_type,
                    "node_a": a,
                    "node_b": b,
                    "similarity": 1.0,
                })

                continue

            # Sequence similarity
            similarity = SequenceMatcher(
                None,
                a_name,
                b_name
            ).ratio()

            if similarity >= 0.92:

                similar_candidates.append({
                    "reason": "very_similar_name",
                    "type": entity_type,
                    "node_a": a,
                    "node_b": b,
                    "similarity": round(similarity, 3),
                })


# ============================================================
# 4. SUSPICIOUS RELATIONSHIPS
# ============================================================

suspicious_edges = []

for edge in edges:

    source = node_by_id.get(edge["source"])
    target = node_by_id.get(edge["target"])

    if not source or not target:
        continue

    relation = edge["relation"]

    if relation not in EXPECTED_TYPES:
        continue

    expected = EXPECTED_TYPES[relation]

    source_type = source["type"]
    target_type = target["type"]

    source_ok = source_type in expected["source"]
    target_ok = target_type in expected["target"]

    if not source_ok or not target_ok:

        suspicious_edges.append({
            "reason": "unexpected_endpoint_type",
            "source": {
                "id": source["id"],
                "name": source["name"],
                "type": source_type,
            },
            "relation": relation,
            "target": {
                "id": target["id"],
                "name": target["name"],
                "type": target_type,
            },
        })


# ============================================================
# 5. EVENTS USED AS JOB ROLES
# ============================================================

event_as_role = []

for edge in edges:

    if edge["relation"] != "HAS_ROLE":
        continue

    target = node_by_id.get(edge["target"])

    if not target:
        continue

    if target["type"] not in {"JobRole", "Role"}:

        event_as_role.append({
            "source": node_by_id[edge["source"]],
            "relation": edge["relation"],
            "target": target,
        })


# ============================================================
# 6. JOB ROLES THAT LOOK LIKE EVENTS
# ============================================================

jobrole_suspicious_names = [
    "tournament",
    "championship",
    "conference",
    "seminar",
    "workshop",
    "competition",
    "festival",
    "meet",
    "program",
    "event",
    "day",
]

suspicious_jobroles = []

for node in nodes:

    if node["type"] != "JobRole":
        continue

    name = node["name"].casefold()

    matched_words = [
        word
        for word in jobrole_suspicious_names
        if word in name
    ]

    if matched_words:

        suspicious_jobroles.append({
            "node": node,
            "matched_keywords": matched_words,
        })


# ============================================================
# 7. ORPHAN NODES
#
# Nodes with no incoming and no outgoing edges.
# ============================================================

connected_nodes = set()

for edge in edges:

    connected_nodes.add(edge["source"])
    connected_nodes.add(edge["target"])


orphan_nodes = []

for node in nodes:

    if node["id"] not in connected_nodes:

        orphan_nodes.append({
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
        })


# ============================================================
# 8. HIGH-DEGREE NODES
# ============================================================

degree = Counter()

for edge in edges:

    degree[edge["source"]] += 1
    degree[edge["target"]] += 1


high_degree_nodes = []

for node_id, count in degree.most_common(50):

    node = node_by_id[node_id]

    high_degree_nodes.append({
        "id": node_id,
        "name": node["name"],
        "type": node["type"],
        "degree": count,
    })


# ============================================================
# 9. RELATIONSHIP TYPE / ENDPOINT TYPE SUMMARY
# ============================================================

relationship_patterns = Counter()

for edge in edges:

    source = node_by_id.get(edge["source"])
    target = node_by_id.get(edge["target"])

    if not source or not target:
        continue

    relationship_patterns[
        (
            edge["relation"],
            source["type"],
            target["type"]
        )
    ] += 1


relationship_pattern_report = []

for (
    relation,
    source_type,
    target_type
), count in relationship_patterns.most_common():

    relationship_pattern_report.append({
        "relation": relation,
        "source_type": source_type,
        "target_type": target_type,
        "count": count,
    })


# ============================================================
# 10. REPORT
# ============================================================

report = {

    "graph_summary": {
        "nodes": len(nodes),
        "edges": len(edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
    },

    "generic_nodes": {
        "count": len(generic_nodes),
        "items": generic_nodes,
    },

    "possible_duplicate_names": {
        "count": len(similar_candidates),
        "items": similar_candidates,
    },

    "suspicious_relationships": {
        "count": len(suspicious_edges),
        "items": suspicious_edges,
    },

    "events_used_as_roles": {
        "count": len(event_as_role),
        "items": event_as_role,
    },

    "suspicious_jobroles": {
        "count": len(suspicious_jobroles),
        "items": suspicious_jobroles,
    },

    "orphan_nodes": {
        "count": len(orphan_nodes),
        "items": orphan_nodes,
    },

    "high_degree_nodes": high_degree_nodes,

    "relationship_patterns": relationship_pattern_report,
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
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
# CONSOLE SUMMARY
# ============================================================

print("=" * 80)
print("NMAMIT GRAPH AUDIT")
print("=" * 80)

print(f"Nodes                    : {len(nodes)}")
print(f"Edges                    : {len(edges)}")

print()
print(f"Generic nodes             : {len(generic_nodes)}")
print(f"Possible duplicate names  : {len(similar_candidates)}")
print(f"Suspicious relationships  : {len(suspicious_edges)}")
print(f"Events used as roles      : {len(event_as_role)}")
print(f"Suspicious JobRoles       : {len(suspicious_jobroles)}")
print(f"Orphan nodes              : {len(orphan_nodes)}")

print()
print("Top high-degree nodes:")

for item in high_degree_nodes[:15]:

    print(
        f"  {item['degree']:>4}  "
        f"{item['name']} "
        f"[{item['type']}]"
    )

print()
print("Top suspicious relationships:")

for item in suspicious_edges[:20]:

    print(
        f"  {item['source']['name']} "
        f"[{item['source']['type']}] "
        f"--{item['relation']}--> "
        f"{item['target']['name']} "
        f"[{item['target']['type']}]"
    )

print()
print(f"Full report: {OUTPUT_FILE}")

print("=" * 80)
