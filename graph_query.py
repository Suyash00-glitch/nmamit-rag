import json
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher


# ============================================================
# CONFIG
# ============================================================

NODES_FILE = "graph_nodes_normalized.json"
EDGES_FILE = "graph_edges_normalized.json"


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
    """

    text = text.casefold()

    # Remove common titles
    text = re.sub(
        r"\b(dr|prof|mr|mrs|ms|sri)\.?\b",
        "",
        text
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# ENTITY ALIASES
# ============================================================

ALIASES = {

    # Institute
    "nmamit": "NMAM Institute of Technology",
    "nmam it": "NMAM Institute of Technology",
    "nmam institute": "NMAM Institute of Technology",
    "nmam institute of technology": "NMAM Institute of Technology",

    # ISE
    "ise": "Information Science & Engineering",
    "information science": "Information Science & Engineering",
    "information science engineering":
        "Information Science & Engineering",

    # AI
    "ai ml": "Artificial Intelligence & Machine Learning",
    "aiml": "Artificial Intelligence & Machine Learning",

    # NSS
    "nss": "National Service Scheme (NSS) Unit",
    "national service scheme":
        "National Service Scheme (NSS) Unit",
}


# ============================================================
# NODE MATCHING
# ============================================================

def exact_matches(query):
    """
    Find nodes whose normalized name exactly matches query.
    """

    q = normalize_text(query)

    results = []

    for node in nodes:

        node_name = normalize_text(node["name"])

        if node_name == q:
            results.append(node)

    return results


def substring_matches(query):
    """
    Find nodes where query occurs inside the node name.
    """

    q = normalize_text(query)

    results = []

    for node in nodes:

        node_name = normalize_text(node["name"])

        if q in node_name:
            results.append(node)

    return results


def fuzzy_matches(query, threshold=0.65):
    """
    Find approximately matching node names.
    """

    q = normalize_text(query)

    scored = []

    for node in nodes:

        node_name = normalize_text(node["name"])

        ratio = SequenceMatcher(
            None,
            q,
            node_name
        ).ratio()

        if ratio >= threshold:
            scored.append(
                (ratio, node)
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored


def find_nodes(query, node_type=None):
    """
    Smart node search.

    Priority:

    1. Alias
    2. Exact match
    3. Substring match
    4. Fuzzy match
    """

    normalized_query = normalize_text(query)

    # --------------------------------------------------------
    # Alias
    # --------------------------------------------------------

    if normalized_query in ALIASES:

        alias_name = ALIASES[normalized_query]

        matches = exact_matches(alias_name)

        if node_type:

            matches = [
                n for n in matches
                if n["type"].casefold()
                == node_type.casefold()
            ]

        if matches:
            return matches

    # --------------------------------------------------------
    # Exact
    # --------------------------------------------------------

    matches = exact_matches(query)

    if node_type:

        matches = [
            n for n in matches
            if n["type"].casefold()
            == node_type.casefold()
        ]

    if matches:
        return matches

    # --------------------------------------------------------
    # Substring
    # --------------------------------------------------------

    matches = substring_matches(query)

    if node_type:

        matches = [
            n for n in matches
            if n["type"].casefold()
            == node_type.casefold()
        ]

    if matches:
        return matches

    # --------------------------------------------------------
    # Fuzzy
    # --------------------------------------------------------

    fuzzy = fuzzy_matches(query)

    results = [
        node
        for score, node in fuzzy
        if (
            not node_type
            or node["type"].casefold()
            == node_type.casefold()
        )
    ]

    return results


# ============================================================
# DISPLAY NODE
# ============================================================

def print_node(node):

    print()
    print("=" * 80)
    print(f"NAME : {node['name']}")
    print(f"TYPE : {node['type']}")
    print(f"ID   : {node['id']}")
    print("=" * 80)


# ============================================================
# GET OUTGOING RELATIONSHIPS
# ============================================================

def get_outgoing(node_id, relation=None):

    results = []

    for edge in edges_from[node_id]:

        if relation:

            if edge["relation"].casefold() != relation.casefold():
                continue

        target = node_by_id.get(edge["target"])

        if target:
            results.append(
                {
                    "relation": edge["relation"],
                    "node": target
                }
            )

    return results


# ============================================================
# GET INCOMING RELATIONSHIPS
# ============================================================

def get_incoming(node_id, relation=None):

    results = []

    for edge in edges_to[node_id]:

        if relation:

            if edge["relation"].casefold() != relation.casefold():
                continue

        source = node_by_id.get(edge["source"])

        if source:
            results.append(
                {
                    "relation": edge["relation"],
                    "node": source
                }
            )

    return results


# ============================================================
# SHOW CONNECTIONS
# ============================================================

def show_connections(
    node,
    relation=None,
    direction="both"
):

    print_node(node)

    # --------------------------------------------------------
    # OUTGOING
    # --------------------------------------------------------

    if direction in ("out", "both"):

        outgoing = get_outgoing(
            node["id"],
            relation
        )

        print("\nOUTGOING:")

        if not outgoing:
            print("  None")

        for item in outgoing:

            target = item["node"]

            print(
                f"  --{item['relation']}--> "
                f"{target['name']} "
                f"[{target['type']}]"
            )

    # --------------------------------------------------------
    # INCOMING
    # --------------------------------------------------------

    if direction in ("in", "both"):

        incoming = get_incoming(
            node["id"],
            relation
        )

        print("\nINCOMING:")

        if not incoming:
            print("  None")

        for item in incoming:

            source = item["node"]

            print(
                f"  <--{item['relation']}-- "
                f"{source['name']} "
                f"[{source['type']}]"
            )


# ============================================================
# ONE-HOP NEIGHBORS
# ============================================================

def get_neighbors(node_id):

    neighbors = []

    # Outgoing
    for edge in edges_from[node_id]:

        target = node_by_id.get(edge["target"])

        if target:
            neighbors.append(
                (
                    edge["relation"],
                    target,
                    "out"
                )
            )

    # Incoming
    for edge in edges_to[node_id]:

        source = node_by_id.get(edge["source"])

        if source:
            neighbors.append(
                (
                    edge["relation"],
                    source,
                    "in"
                )
            )

    return neighbors


# ============================================================
# BFS PATH
# ============================================================

def shortest_path(start_id, end_id):

    queue = deque([start_id])

    visited = {
        start_id
    }

    parent = {}

    while queue:

        current = queue.popleft()

        if current == end_id:
            break

        # Outgoing
        for edge in edges_from[current]:

            next_id = edge["target"]

            if next_id not in visited:

                visited.add(next_id)

                parent[next_id] = (
                    current,
                    edge
                )

                queue.append(next_id)

        # Incoming
        for edge in edges_to[current]:

            next_id = edge["source"]

            if next_id not in visited:

                visited.add(next_id)

                parent[next_id] = (
                    current,
                    edge
                )

                queue.append(next_id)

    # No path
    if end_id not in visited:
        return None

    # Reconstruct path
    path = []

    current = end_id

    while current != start_id:

        previous, edge = parent[current]

        path.append(
            (
                previous,
                edge,
                current
            )
        )

        current = previous

    path.reverse()

    return path


# ============================================================
# PRINT PATH
# ============================================================

def print_path(path):

    if not path:

        print("No path found.")

        return

    print("\nPATH")
    print("-" * 80)

    for source_id, edge, target_id in path:

        source = node_by_id[source_id]
        target = node_by_id[target_id]

        print(
            f"{source['name']}"
            f" [{source['type']}]"
        )

        print(
            f"   --{edge['relation']}-->"
        )

    target = node_by_id[path[-1][2]]

    print(
        f"{target['name']}"
        f" [{target['type']}]"
    )


# ============================================================
# GRAPH CONTEXT
# ============================================================

def generate_context(
    node,
    max_connections=20
):

    lines = []

    lines.append(
        f"Entity: {node['name']}"
    )

    lines.append(
        f"Type: {node['type']}"
    )

    # Outgoing
    outgoing = get_outgoing(node["id"])

    for item in outgoing[:max_connections]:

        target = item["node"]

        lines.append(
            f"{node['name']} "
            f"--{item['relation']}--> "
            f"{target['name']} "
            f"[{target['type']}]"
        )

    # Incoming
    incoming = get_incoming(node["id"])

    for item in incoming[:max_connections]:

        source = item["node"]

        lines.append(
            f"{source['name']} "
            f"--{item['relation']}--> "
            f"{node['name']} "
            f"[{source['type']}]"
        )

    return "\n".join(lines)


# ============================================================
# SEARCH TEST
# ============================================================

def test_search(query):

    print()
    print("#" * 80)
    print(f"SEARCH: {query}")
    print("#" * 80)

    matches = find_nodes(query)

    print(
        f"Matches: {len(matches)}"
    )

    for node in matches[:10]:

        print(
            f"  {node['name']} "
            f"[{node['type']}] "
            f"({node['id']})"
        )


# ============================================================
# RELATION QUERY
# ============================================================

def relation_query(
    entity_name,
    relation=None,
    direction="out"
):

    matches = find_nodes(entity_name)

    if not matches:

        print(
            f"No entity found for: "
            f"{entity_name}"
        )

        return

    # Usually first/best result
    node = matches[0]

    print()
    print(
        f"ENTITY: {node['name']}"
    )

    print(
        f"TYPE: {node['type']}"
    )

    print()

    connections = []

    if direction in ("out", "both"):

        connections.extend(
            [
                (
                    item["relation"],
                    item["node"],
                    "out"
                )
                for item in get_outgoing(
                    node["id"],
                    relation
                )
            ]
        )

    if direction in ("in", "both"):

        connections.extend(
            [
                (
                    item["relation"],
                    item["node"],
                    "in"
                )
                for item in get_incoming(
                    node["id"],
                    relation
                )
            ]
        )

    if not connections:

        print("No matching relationships.")

        return

    for rel, target, direction_type in connections:

        if direction_type == "out":

            print(
                f"--{rel}--> "
                f"{target['name']} "
                f"[{target['type']}]"
            )

        else:

            print(
                f"<--{rel}-- "
                f"{target['name']} "
                f"[{target['type']}]"
            )


# ============================================================
# PATH QUERY
# ============================================================

def path_query(
    start_name,
    end_name
):

    start_matches = find_nodes(start_name)
    end_matches = find_nodes(end_name)

    if not start_matches:

        print(
            f"Start entity not found: "
            f"{start_name}"
        )

        return

    if not end_matches:

        print(
            f"End entity not found: "
            f"{end_name}"
        )

        return

    start = start_matches[0]
    end = end_matches[0]

    print()
    print(
        f"FROM: {start['name']}"
    )

    print(
        f"TO  : {end['name']}"
    )

    path = shortest_path(
        start["id"],
        end["id"]
    )

    print_path(path)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive():

    print()
    print("=" * 80)
    print("NMAMIT GRAPH QUERY ENGINE")
    print("=" * 80)

    print(
        f"Nodes : {len(nodes)}"
    )

    print(
        f"Edges : {len(edges)}"
    )

    print()
    print("Commands:")
    print()
    print("  search <name>")
    print("  show <name>")
    print("  out <name>")
    print("  in <name>")
    print("  relation <name> <RELATION>")
    print("  path <start> -> <end>")
    print("  context <name>")
    print("  quit")
    print()

    while True:

        try:
            command = input("graph> ").strip()

        except (KeyboardInterrupt, EOFError):

            print()
            break

        if not command:
            continue

        if command.casefold() in (
            "quit",
            "exit",
            "q"
        ):
            break

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if command.casefold().startswith("search "):

            query = command[7:].strip()

            test_search(query)

            continue

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        if command.casefold().startswith("show "):

            query = command[5:].strip()

            matches = find_nodes(query)

            if not matches:

                print("No matching nodes.")

                continue

            show_connections(
                matches[0]
            )

            continue

        # ----------------------------------------------------
        # OUT
        # ----------------------------------------------------

        if command.casefold().startswith("out "):

            query = command[4:].strip()

            relation_query(
                query,
                direction="out"
            )

            continue

        # ----------------------------------------------------
        # IN
        # ----------------------------------------------------

        if command.casefold().startswith("in "):

            query = command[3:].strip()

            relation_query(
                query,
                direction="in"
            )

            continue

        # ----------------------------------------------------
        # RELATION
        # ----------------------------------------------------

        if command.casefold().startswith("relation "):

            parts = command.split()

            if len(parts) < 3:

                print(
                    "Usage: "
                    "relation <name> <RELATION>"
                )

                continue

            relation = parts[-1]

            entity = " ".join(
                parts[1:-1]
            )

            relation_query(
                entity,
                relation=relation,
                direction="both"
            )

            continue

        # ----------------------------------------------------
        # PATH
        # ----------------------------------------------------

        if command.casefold().startswith("path "):

            expression = command[5:]

            if "->" not in expression:

                print(
                    "Usage: "
                    "path <start> -> <end>"
                )

                continue

            start, end = expression.split(
                "->",
                1
            )

            path_query(
                start.strip(),
                end.strip()
            )

            continue

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        if command.casefold().startswith("context "):

            query = command[8:].strip()

            matches = find_nodes(query)

            if not matches:

                print("No matching node.")

                continue

            print()

            print(
                generate_context(
                    matches[0]
                )
            )

            continue

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        print(
            "Unknown command."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactive()
