import json
from collections import defaultdict

NODES_FILE = "graph_nodes_resolved.json"
EDGES_FILE = "graph_edges_resolved.json"


def load_graph():
    with open(NODES_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(EDGES_FILE, "r", encoding="utf-8") as f:
        edges = json.load(f)

    return nodes, edges


def get_value(obj, *keys):
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def search(nodes, query):
    query = query.lower()

    matches = []

    for node in nodes:
        name = get_value(node, "name", "label", "entity")

        if name and query in name.lower():
            matches.append(node)

    return matches


def show(node, edges, nodes_by_id):
    node_id = get_value(node, "id", "node_id")
    name = get_value(node, "name", "label", "entity")
    node_type = get_value(node, "type", "entity_type")

    print("\n" + "=" * 80)
    print(f"NODE: {name}")
    print(f"TYPE: {node_type}")
    print(f"ID  : {node_id}")
    print("=" * 80)

    print("\nOUTGOING:")

    for edge in edges:
        source = get_value(edge, "source", "from")
        target = get_value(edge, "target", "to")
        relation = get_value(edge, "relation", "type")

        if source == node_id:
            target_node = nodes_by_id.get(target)

            if target_node:
                target_name = get_value(
                    target_node, "name", "label", "entity"
                )
                target_type = get_value(
                    target_node, "type", "entity_type"
                )

                print(
                    f"  --{relation}--> "
                    f"{target_name} [{target_type}]"
                )

    print("\nINCOMING:")

    for edge in edges:
        source = get_value(edge, "source", "from")
        target = get_value(edge, "target", "to")
        relation = get_value(edge, "relation", "type")

        if target == node_id:
            source_node = nodes_by_id.get(source)

            if source_node:
                source_name = get_value(
                    source_node, "name", "label", "entity"
                )
                source_type = get_value(
                    source_node, "type", "entity_type"
                )

                print(
                    f"  <--{relation}-- "
                    f"{source_name} [{source_type}]"
                )


def main():
    nodes, edges = load_graph()

    nodes_by_id = {}

    for node in nodes:
        node_id = get_value(node, "id", "node_id")

        if node_id:
            nodes_by_id[node_id] = node

    print("=" * 80)
    print("NMAMIT RESOLVED GRAPH TEST")
    print("=" * 80)

    print(f"Nodes : {len(nodes)}")
    print(f"Edges : {len(edges)}")

    queries = [
        "NMAM Institute of Technology",
        "Information Science & Engineering",
        "Niranjan",
        "National Service Scheme",
        "Research and Innovation Center",
        "Artificial Intelligence"
    ]

    for query in queries:

        print("\n" + "#" * 80)
        print(f"SEARCH: {query}")
        print("#" * 80)

        matches = search(nodes, query)

        print(f"Matches: {len(matches)}")

        for node in matches[:10]:
            name = get_value(node, "name", "label", "entity")
            node_type = get_value(node, "type", "entity_type")
            node_id = get_value(node, "id", "node_id")

            print(
                f"\n{name} [{node_type}] ({node_id})"
            )

            show(node, edges, nodes_by_id)

    print("\n" + "=" * 80)
    print("RESOLVED GRAPH TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
