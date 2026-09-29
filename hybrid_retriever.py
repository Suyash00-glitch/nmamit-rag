import json
import re
from collections import defaultdict


# ============================================================
# FILES
# ============================================================

NODES_FILE = "graph_nodes_resolved.json"
EDGES_FILE = "graph_edges_resolved.json"
CHUNKS_FILE = "chunks.json"


# ============================================================
# LOAD DATA
# ============================================================

with open(NODES_FILE, "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open(EDGES_FILE, "r", encoding="utf-8") as f:
    edges = json.load(f)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)


# ============================================================
# INDEX GRAPH
# ============================================================

nodes_by_id = {}
name_index = defaultdict(list)

for node in nodes:
    node_id = node["id"]
    name = node["name"]

    nodes_by_id[node_id] = node

    # normalized name
    normalized = re.sub(r"\s+", " ", name.lower().strip())

    name_index[normalized].append(node_id)


outgoing = defaultdict(list)
incoming = defaultdict(list)

for edge in edges:
    outgoing[edge["source"]].append(edge)
    incoming[edge["target"]].append(edge)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s&.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# ENTITY SEARCH
# ============================================================

def search_entities(query, limit=10):
    """
    Find graph entities relevant to the query.

    Exact matches are preferred.
    Partial/token matches are used afterwards.
    """

    q = normalize(query)

    results = []

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for name, ids in name_index.items():

        if q == name:
            for node_id in ids:
                results.append(
                    (100, nodes_by_id[node_id])
                )

    # --------------------------------------------------------
    # Name contains query
    # --------------------------------------------------------

    for name, ids in name_index.items():

        if q in name and q != name:

            for node_id in ids:
                results.append(
                    (80, nodes_by_id[node_id])
                )

    # --------------------------------------------------------
    # Query contains entity name
    # --------------------------------------------------------

    for name, ids in name_index.items():

        if name in q and name != q:

            for node_id in ids:
                results.append(
                    (70, nodes_by_id[node_id])
                )

    # --------------------------------------------------------
    # Token matching
    # --------------------------------------------------------

    query_tokens = set(q.split())

    for name, ids in name_index.items():

        name_tokens = set(name.split())

        if not name_tokens:
            continue

        overlap = len(query_tokens & name_tokens)

        if overlap > 0:

            score = 40 + (
                overlap / len(query_tokens)
            ) * 30

            for node_id in ids:
                results.append(
                    (score, nodes_by_id[node_id])
                )

    # --------------------------------------------------------
    # Remove duplicate nodes
    # --------------------------------------------------------

    best = {}

    for score, node in results:

        node_id = node["id"]

        if node_id not in best:
            best[node_id] = score
        else:
            best[node_id] = max(
                best[node_id],
                score
            )

    ranked = sorted(
        best.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        nodes_by_id[node_id]
        for node_id, score in ranked[:limit]
    ]


# ============================================================
# GRAPH RELATIONSHIPS
# ============================================================

def get_relationships(node_ids, max_relationships=30):
    """
    Retrieve relationships connected to the selected entities.

    We keep the result bounded so the LLM doesn't receive
    thousands of irrelevant relationships.
    """

    node_ids = set(node_ids)

    relationships = []

    for node_id in node_ids:

        # ----------------------------------------------------
        # Outgoing
        # ----------------------------------------------------

        for edge in outgoing.get(node_id, []):

            relationships.append({
                "source": nodes_by_id[edge["source"]],
                "relation": edge["relation"],
                "target": nodes_by_id[edge["target"]],
                "direction": "outgoing"
            })

        # ----------------------------------------------------
        # Incoming
        # ----------------------------------------------------

        for edge in incoming.get(node_id, []):

            relationships.append({
                "source": nodes_by_id[edge["source"]],
                "relation": edge["relation"],
                "target": nodes_by_id[edge["target"]],
                "direction": "incoming"
            })

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}

    for rel in relationships:

        key = (
            rel["source"]["id"],
            rel["relation"],
            rel["target"]["id"]
        )

        unique[key] = rel

    relationships = list(unique.values())

    # --------------------------------------------------------
    # Prioritize useful relations
    # --------------------------------------------------------

    priority = {
        "OFFERS": 10,
        "HAS_PROGRAM": 10,
        "HAS_FACULTY": 9,
        "WORKS_AT": 9,
        "COLLABORATED_WITH": 8,
        "HAS_FACILITY": 7,
        "HAS_PLACEMENT": 10,
        "LOCATED_IN": 5,
        "PARTICIPATED_IN": 4,
        "HAS_ROLE": 6,
        "WORKS_ON": 5,
    }

    relationships.sort(
        key=lambda r: priority.get(
            r["relation"],
            1
        ),
        reverse=True
    )

    return relationships[:max_relationships]


# ============================================================
# SIMPLE KEYWORD CHUNK RETRIEVAL
# ============================================================

def retrieve_chunks(query, top_k=5):
    """
    Simple lexical retrieval from chunks.json.

    This is intentionally lightweight for now.

    Later we can replace this with embeddings + FAISS/Chroma.
    """

    query_tokens = set(
        normalize(query).split()
    )

    scored = []

    for chunk in chunks:

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        if isinstance(chunk, dict):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )

        else:
            text = str(chunk)

        if not text:
            continue

        text_normalized = normalize(text)

        text_tokens = set(
            text_normalized.split()
        )

        # ----------------------------------------------------
        # Keyword overlap
        # ----------------------------------------------------

        overlap = query_tokens & text_tokens

        if not overlap:
            continue

        score = len(overlap)

        # Slight boost for exact phrase
        if normalize(query) in text_normalized:
            score += 5

        scored.append(
            (score, chunk)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored[:top_k]
    ]


# ============================================================
# HYBRID RETRIEVER
# ============================================================

def retrieve(
    query,
    entity_limit=5,
    max_relationships=30,
    chunk_top_k=5
):

    # ========================================================
    # 1. GRAPH ENTITY RETRIEVAL
    # ========================================================

    entities = search_entities(
        query,
        limit=entity_limit
    )

    entity_ids = [
        entity["id"]
        for entity in entities
    ]

    # ========================================================
    # 2. GRAPH RELATIONSHIPS
    # ========================================================

    relationships = get_relationships(
        entity_ids,
        max_relationships=max_relationships
    )

    # ========================================================
    # 3. DOCUMENT RETRIEVAL
    # ========================================================

    relevant_chunks = retrieve_chunks(
        query,
        top_k=chunk_top_k
    )

    return {
        "query": query,
        "entities": entities,
        "relationships": relationships,
        "chunks": relevant_chunks
    }


# ============================================================
# DISPLAY
# ============================================================

def print_results(result):

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL")
    print("=" * 70)

    print()
    print("QUERY:")
    print(result["query"])

    # ========================================================
    # ENTITIES
    # ========================================================

    print()
    print("=" * 70)
    print("RELEVANT ENTITIES")
    print("=" * 70)

    for entity in result["entities"]:

        print(
            f"- {entity['name']} "
            f"[{entity['type']}] "
            f"({entity['id']})"
        )

    # ========================================================
    # GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("GRAPH RELATIONSHIPS")
    print("=" * 70)

    for rel in result["relationships"]:

        source = rel["source"]
        target = rel["target"]

        print(
            f"- {source['name']} "
            f"[{source['type']}] "
            f"--{rel['relation']}--> "
            f"{target['name']} "
            f"[{target['type']}]"
        )

    # ========================================================
    # DOCUMENT CHUNKS
    # ========================================================

    print()
    print("=" * 70)
    print("DOCUMENT CHUNKS")
    print("=" * 70)

    for i, chunk in enumerate(
        result["chunks"],
        start=1
    ):

        print()
        print(f"CHUNK {i}")
        print("-" * 70)

        if isinstance(chunk, dict):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )

            print(text)

            # Show metadata if available

            metadata = chunk.get(
                "metadata"
            )

            if metadata:
                print()
                print("METADATA:")
                print(metadata)

        else:

            print(chunk)


# ============================================================
# TERMINAL
# ============================================================

def main():

    print("=" * 70)
    print("NMAMIT HYBRID GRAPH + DOCUMENT RETRIEVER")
    print("=" * 70)

    print()
    print(f"Nodes  : {len(nodes)}")
    print(f"Edges  : {len(edges)}")
    print(f"Chunks : {len(chunks)}")

    print()
    print("Commands:")
    print()
    print("  retrieve <query>")
    print("  quit")
    print()

    while True:

        try:
            command = input("retriever> ").strip()

        except KeyboardInterrupt:
            print()
            break

        if not command:
            continue

        if command.lower() == "quit":
            break

        if command.lower().startswith(
            "retrieve "
        ):

            query = command[
                len("retrieve "):
            ].strip()

            if not query:
                print("Enter a query.")
                continue

            result = retrieve(query)

            print_results(result)

        else:

            print(
                "Unknown command. "
                "Use: retrieve <query>"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
