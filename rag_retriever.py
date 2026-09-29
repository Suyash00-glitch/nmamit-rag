import json
import re
from collections import defaultdict, deque


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
# INDEXES
# ============================================================

nodes_by_id = {}
nodes_by_name = defaultdict(list)

for node in nodes:
    node_id = node["id"]
    name = node["name"]

    nodes_by_id[node_id] = node
    nodes_by_name[name.lower()].append(node_id)


# Graph adjacency
outgoing = defaultdict(list)
incoming = defaultdict(list)

for edge in edges:
    source = edge["source"]
    target = edge["target"]

    outgoing[source].append(edge)
    incoming[target].append(edge)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# ENTITY SEARCH
# ============================================================

def search_entities(query, limit=10):

    query_norm = normalize(query)
    query_words = set(query_norm.split())

    results = []

    for node in nodes:

        name_norm = normalize(node["name"])
        name_words = set(name_norm.split())

        score = 0

        # Exact match
        if query_norm == name_norm:
            score += 100

        # Query contained in entity name
        elif query_norm in name_norm:
            score += 60

        # Entity name contained in query
        elif name_norm in query_norm:
            score += 50

        # Word overlap
        overlap = query_words.intersection(name_words)
        score += len(overlap) * 10

        if score > 0:
            results.append((score, node))

    results.sort(key=lambda x: x[0], reverse=True)

    return results[:limit]


# ============================================================
# GET NEIGHBORS
# ============================================================

def get_neighbors(node_id, depth=1):

    visited = {node_id}
    queue = deque([(node_id, 0)])

    collected = []

    while queue:

        current, current_depth = queue.popleft()

        if current_depth >= depth:
            continue

        # OUTGOING
        for edge in outgoing[current]:

            target = edge["target"]

            collected.append({
                "direction": "outgoing",
                "source": nodes_by_id[current],
                "relation": edge["relation"],
                "target": nodes_by_id[target]
            })

            if target not in visited:
                visited.add(target)
                queue.append((target, current_depth + 1))

        # INCOMING
        for edge in incoming[current]:

            source = edge["source"]

            collected.append({
                "direction": "incoming",
                "source": nodes_by_id[source],
                "relation": edge["relation"],
                "target": nodes_by_id[current]
            })

            if source not in visited:
                visited.add(source)
                queue.append((source, current_depth + 1))

    return collected


# ============================================================
# RETRIEVE GRAPH CONTEXT
# ============================================================

def retrieve(query, entity_limit=5, depth=1):

    matches = search_entities(query, entity_limit)

    if not matches:
        return {
            "query": query,
            "entities": [],
            "relationships": []
        }

    entity_ids = set()
    entities = []

    for score, node in matches:

        entity_ids.add(node["id"])

        entities.append({
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "score": score
        })

    relationships = []

    seen_edges = set()

    for node_id in entity_ids:

        neighbors = get_neighbors(node_id, depth)

        for item in neighbors:

            source_id = item["source"]["id"]
            target_id = item["target"]["id"]
            relation = item["relation"]

            edge_key = (
                source_id,
                relation,
                target_id
            )

            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)

            relationships.append({
                "source": item["source"]["name"],
                "source_type": item["source"]["type"],
                "relation": relation,
                "target": item["target"]["name"],
                "target_type": item["target"]["type"]
            })

    return {
        "query": query,
        "entities": entities,
        "relationships": relationships
    }


# ============================================================
# FORMAT FOR LLM
# ============================================================

def format_context(result):

    lines = []

    lines.append("GRAPH CONTEXT")
    lines.append("=" * 60)

    lines.append("\nRELEVANT ENTITIES:")

    for entity in result["entities"]:

        lines.append(
            f"- {entity['name']} "
            f"[{entity['type']}]"
        )

    lines.append("\nRELATIONSHIPS:")

    for rel in result["relationships"]:

        lines.append(
            f"- {rel['source']} "
            f"[{rel['source_type']}] "
            f"--{rel['relation']}--> "
            f"{rel['target']} "
            f"[{rel['target_type']}]"
        )

    return "\n".join(lines)


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NMAMIT GRAPH RAG RETRIEVER")
    print("=" * 70)

    print(f"Nodes : {len(nodes)}")
    print(f"Edges : {len(edges)}")

    print("\nCommands:")
    print("  retrieve <query>")
    print("  search <query>")
    print("  quit")

    while True:

        try:
            command = input("\nretriever> ").strip()

        except KeyboardInterrupt:
            break

        if not command:
            continue

        if command.lower() == "quit":
            break

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if command.lower().startswith("search "):

            query = command[7:].strip()

            results = search_entities(query)

            print("\nSEARCH RESULTS")
            print("-" * 60)

            for score, node in results:

                print(
                    f"{node['name']} "
                    f"[{node['type']}] "
                    f"score={score}"
                )

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        elif command.lower().startswith("retrieve "):

            query = command[9:].strip()

            result = retrieve(
                query,
                entity_limit=5,
                depth=1
            )

            print()

            print(format_context(result))

        else:

            print("Unknown command.")
