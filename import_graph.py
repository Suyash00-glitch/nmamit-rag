import json
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

NODES_FILE = "graph_nodes_resolved.json"
EDGES_FILE = "graph_edges_resolved.json"
BATCH_SIZE = 500


def check_config():
    values = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_nodes(session, nodes):
    query = """
    UNWIND $rows AS row
    MERGE (n:Entity {id: row.id})
    SET n.name = row.name,
        n.type = row.type
    """
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        session.run(query, rows=batch).consume()
        print(f"  Nodes imported: {min(i + BATCH_SIZE, len(nodes))}/{len(nodes)}")


def clean_relation_type(relation):
    relation = str(relation).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", relation):
        raise ValueError(f"Invalid relationship type: {relation!r}")
    return relation


def import_edges(session, edges):
    grouped = {}
    for edge in edges:
        relation = clean_relation_type(edge["relation"])
        grouped.setdefault(relation, []).append({
            "source": edge["source"],
            "target": edge["target"],
        })

    total = len(edges)
    imported = 0

    for relation, rows in grouped.items():
        query = f"""
        UNWIND $rows AS row
        MATCH (source:Entity {{id: row.source}})
        MATCH (target:Entity {{id: row.target}})
        MERGE (source)-[:{relation}]->(target)
        """
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.run(query, rows=batch).consume()
            imported += len(batch)
            print(f"  Relationships imported: {imported}/{total} ({relation})")


def verify(session):
    print("\n" + "=" * 70)
    print("VERIFYING AURADB")
    print("=" * 70)

    node_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
    edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    print(f"Nodes         : {node_count}")
    print(f"Relationships : {edge_count}")

    print("\nSample nodes:")
    for record in session.run("""
        MATCH (n:Entity)
        RETURN n.id AS id, n.name AS name, n.type AS type
        LIMIT 5
    """):
        print(f"  {record['id']} | {record['name']} | {record['type']}")


def main():
    print("=" * 70)
    print("NMAMIT GRAPH -> NEO4J AURADB")
    print("=" * 70)

    check_config()
    nodes = load_json(NODES_FILE)
    edges = load_json(EDGES_FILE)

    print(f"\nNodes found : {len(nodes)}")
    print(f"Edges found : {len(edges)}")
    print("\nConnecting to Neo4j AuraDB...")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()
        print("Connection successful.")

        with driver.session(database=NEO4J_DATABASE) as session:
            print("\n" + "=" * 70)
            print("IMPORTING NODES")
            print("=" * 70)
            import_nodes(session, nodes)

            print("\n" + "=" * 70)
            print("IMPORTING RELATIONSHIPS")
            print("=" * 70)
            import_edges(session, edges)

            verify(session)
    finally:
        driver.close()

    print("\n" + "=" * 70)
    print("IMPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
