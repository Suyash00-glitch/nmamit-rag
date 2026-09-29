import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

INPUT = "graph_nodes.json"


with open(INPUT, "r", encoding="utf-8") as f:
    nodes = json.load(f)


# ---------------------------------------------------------
# Normalization used ONLY for detecting candidates
# ---------------------------------------------------------

def normalize_for_compare(name):
    s = name.casefold().strip()

    # Remove common titles
    s = re.sub(
        r"\b(dr|prof|mr|mrs|ms|sri|shri|capt)\.?\s+",
        "",
        s
    )

    # Remove punctuation
    s = re.sub(r"[^a-z0-9& ]", " ", s)

    # Normalize common formatting
    s = s.replace("&", "and")

    s = re.sub(r"\s+", " ", s).strip()

    return s


# ---------------------------------------------------------
# Group by TYPE
# ---------------------------------------------------------

by_type = defaultdict(list)

for node in nodes:
    by_type[node["type"]].append(node)


# ---------------------------------------------------------
# Find suspicious pairs
# ---------------------------------------------------------

results = []


for node_type, type_nodes in by_type.items():

    for i in range(len(type_nodes)):

        a = type_nodes[i]
        na = normalize_for_compare(a["name"])

        for j in range(i + 1, len(type_nodes)):

            b = type_nodes[j]
            nb = normalize_for_compare(b["name"])

            # Already exactly same after normalization
            if na == nb:
                score = 1.0

            else:
                score = SequenceMatcher(
                    None,
                    na,
                    nb
                ).ratio()

            # Only show strong candidates
            if score >= 0.88:

                results.append({
                    "type": node_type,
                    "score": round(score, 3),
                    "node_a": a,
                    "node_b": b
                })


# ---------------------------------------------------------
# Sort strongest first
# ---------------------------------------------------------

results.sort(
    key=lambda x: (
        x["type"],
        -x["score"]
    )
)


# ---------------------------------------------------------
# Print
# ---------------------------------------------------------

print("=" * 80)
print("CANONICALIZATION CANDIDATES")
print("=" * 80)

print(f"Nodes: {len(nodes)}")
print(f"Candidates: {len(results)}")
print()


for x in results[:300]:

    print(
        f"[{x['type']}] "
        f"score={x['score']}"
    )

    print(
        f"  A: {x['node_a']['name']} "
        f"({x['node_a']['id']})"
    )

    print(
        f"  B: {x['node_b']['name']} "
        f"({x['node_b']['id']})"
    )

    print()


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(
    "canonicalization_candidates.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print("=" * 80)
print("Saved: canonicalization_candidates.json")
print("=" * 80)
