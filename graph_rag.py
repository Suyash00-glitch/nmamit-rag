import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nmamit_chunks")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NORMAL_ENTITY_LIMIT = 8
NORMAL_RELATIONSHIP_LIMIT = 40
NORMAL_VECTOR_LIMIT = 6
EXHAUSTIVE_ENTITY_LIMIT = 50
EXHAUSTIVE_RELATIONSHIP_LIMIT = 300
EXHAUSTIVE_VECTOR_LIMIT = 12
MAX_CHUNK_CHARS = 3500


def check_config():
    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))


def is_exhaustive_query(question):
    q = question.lower()
    phrases = [
        "list all", "all the", "all programs", "all programme", "all programmes",
        "all courses", "all branches", "all departments", "all facilities",
        "all companies", "every program", "every programme", "every course",
        "every branch", "every department", "what programs", "what programmes",
        "what courses", "what branches", "what departments", "which programs",
        "which programmes", "which branches", "which departments",
        "what does nmamit offer", "what does nmamit have"
    ]
    return any(p in q for p in phrases)


def find_entities(session, question, limit):
    words = [w for w in re.findall(r"[a-zA-Z0-9&.-]+", question) if len(w) >= 3]
    if not words:
        return []

    query = """
    MATCH (n:Entity)
    WHERE any(word IN $words WHERE toLower(n.name) CONTAINS toLower(word))
    RETURN n.id AS id, n.name AS name, n.type AS type,
           reduce(score = 0, word IN $words |
               score + CASE
                   WHEN toLower(n.name) = toLower(word) THEN 10
                   WHEN toLower(n.name) CONTAINS toLower(word) THEN 1
                   ELSE 0
               END) AS score
    ORDER BY score DESC, size(n.name) ASC
    LIMIT $limit
    """
    return [r.data() for r in session.run(query, words=words, limit=limit)]


def get_graph_context(session, entities, relationship_limit):
    if not entities:
        return [], []
    ids = [e["id"] for e in entities]
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.id IN $ids OR b.id IN $ids
    RETURN a.name AS source_name, a.type AS source_type,
           type(r) AS relation, b.name AS target_name, b.type AS target_type
    LIMIT $limit
    """
    return entities, [r.data() for r in session.run(query, ids=ids, limit=relationship_limit)]


def get_offered_programs(session):
    """Structured enumeration: don't rely on vector similarity for 'list all' queries."""
    queries = [
        """
        MATCH (college:Entity)-[:OFFERS]->(program:Entity)
        WHERE toLower(college.name) CONTAINS 'nmam'
        RETURN DISTINCT program.id AS id, program.name AS name, program.type AS type
        ORDER BY program.name
        """,
        """
        MATCH (college:Entity)-[:HAS_PROGRAM]->(program:Entity)
        WHERE toLower(college.name) CONTAINS 'nmam'
        RETURN DISTINCT program.id AS id, program.name AS name, program.type AS type
        ORDER BY program.name
        """
    ]
    programs = {}
    for query in queries:
        for r in session.run(query):
            d = r.data()
            programs[d["id"]] = d
    return sorted(programs.values(), key=lambda x: x["name"].lower())


def vector_search(qdrant, embedding_model, question, limit):
    query_vector = embedding_model.encode(question, normalize_embeddings=True).tolist()
    result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    chunks = []
    for point in result.points:
        payload = point.payload or {}
        content = (payload.get("content") or "").strip()
        if not content:
            continue
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + "..."
        chunks.append({
            "score": float(point.score),
            "chunk_id": payload.get("chunk_id"),
            "content": content,
            "page_title": payload.get("page_title"),
            "section": payload.get("section"),
            "source_url": payload.get("source_url"),
            "department": payload.get("department"),
        })
    return chunks


def build_graph_context(entities, relationships, programs=None):
    lines = ["RELEVANT ENTITIES:"]
    for e in entities:
        lines.append(f"- {e['name']} [{e['type']}]")

    if programs:
        lines += ["", "PROGRAMS CONNECTED TO NMAMIT:"]
        for p in programs:
            lines.append(f"- {p['name']} [{p['type']}]")

    lines += ["", "GRAPH RELATIONSHIPS:"]
    for r in relationships:
        lines.append(
            f"- {r['source_name']} [{r['source_type']}] --{r['relation']}--> "
            f"{r['target_name']} [{r['target_type']}]"
        )
    return "\n".join(lines)


def build_vector_context(chunks):
    if not chunks:
        return "No relevant document chunks were retrieved."
    lines = []
    for i, c in enumerate(chunks, 1):
        lines += [
            f"DOCUMENT CHUNK {i}",
            f"Chunk ID: {c['chunk_id']}",
            f"Page: {c['page_title']}",
            f"Section: {c['section']}",
            f"Similarity: {c['score']:.4f}",
            c["content"],
            "",
        ]
    return "\n".join(lines)


def ask_groq(client, question, graph_context, vector_context, exhaustive):
    if exhaustive:
        style = """
This is an exhaustive/list-style question. Give the complete list supported by
retrieved evidence, not a few examples. If a structured PROGRAMS CONNECTED TO
NMAMIT section is present, use it as the authoritative list for that query.
Do not add items merely because you know them from outside knowledge. If the
retrieved graph cannot establish completeness, say so rather than pretending.
"""
    else:
        style = """
Answer naturally with enough detail for the question. For prospective-student
questions such as why someone should join NMAMIT, branch overviews, facilities,
industry exposure, placements, or campus opportunities, give a detailed,
engaging and naturally persuasive answer. Explain why the retrieved facts may
matter to a student instead of merely dumping raw graph relationships.
Persuasion must never override factual grounding. Do not invent rankings,
salary figures, placement statistics, outcomes, facilities, collaborations, or
other advantages not supported by the retrieved evidence.
"""

    system = f"""
You are the NMAMIT information and admissions assistant.

Use ONLY the supplied Neo4j and Qdrant evidence.

GROUNDING RULES:
1. Never invent facts or use outside knowledge.
2. Retrieved graph relationships are evidence, but strange/noisy relationships
   must not be transformed into confident claims beyond what they literally say.
3. Qdrant chunks are source text; synthesize them instead of inventing details.
4. If evidence is insufficient, clearly say so.
5. Do not mention internal retrieval implementation unless asked.
6. For factual list questions, preserve all supported items.
7. For admission/discovery questions, be informative and welcoming, not generic hype.

{style}
"""

    user = f"""QUESTION:\n{question}\n\n=== NEO4J GRAPH EVIDENCE ===\n{graph_context}\n\n=== QDRANT DOCUMENT EVIDENCE ===\n{vector_context}\n"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def retrieve_and_answer(driver, qdrant, embedding_model, groq_client, question):
    exhaustive = is_exhaustive_query(question)
    entity_limit = EXHAUSTIVE_ENTITY_LIMIT if exhaustive else NORMAL_ENTITY_LIMIT
    relationship_limit = EXHAUSTIVE_RELATIONSHIP_LIMIT if exhaustive else NORMAL_RELATIONSHIP_LIMIT
    vector_limit = EXHAUSTIVE_VECTOR_LIMIT if exhaustive else NORMAL_VECTOR_LIMIT

    with driver.session(database=NEO4J_DATABASE) as session:
        entities = find_entities(session, question, entity_limit)
        entities, relationships = get_graph_context(session, entities, relationship_limit)

        programs = []
        q = question.lower()
        if exhaustive and any(x in q for x in ["program", "programme", "branch", "course"]):
            programs = get_offered_programs(session)

    vector_chunks = vector_search(qdrant, embedding_model, question, vector_limit)
    graph_context = build_graph_context(entities, relationships, programs)
    vector_context = build_vector_context(vector_chunks)

    print("\n" + "=" * 70)
    print("NEO4J GRAPH CONTEXT")
    print("=" * 70)
    print(graph_context)

    print("\n" + "=" * 70)
    print("QDRANT DOCUMENT CONTEXT")
    print("=" * 70)
    for i, c in enumerate(vector_chunks, 1):
        print(f"\nCHUNK {i}")
        print("-" * 70)
        print(c["content"])

    print("\n" + "=" * 70)
    print("GROQ ANSWER")
    print("=" * 70)
    print(ask_groq(groq_client, question, graph_context, vector_context, exhaustive))


def main():
    check_config()

    print("=" * 70)
    print("NMAMIT HYBRID GRAPH RAG")
    print("Neo4j AuraDB + Qdrant Cloud + Groq")
    print("=" * 70)

    print("\nConnecting to Neo4j AuraDB...")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("Neo4j connection successful.")

    print("\nConnecting to Qdrant Cloud...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info = qdrant.get_collection(QDRANT_COLLECTION)
    print("Qdrant connection successful.")
    print("Collection:", QDRANT_COLLECTION)
    print("Points:", info.points_count)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
    print("Vector dimension:", embedding_model.get_sentence_embedding_dimension())

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client ready.")
    print("Groq model:", GROQ_MODEL)

    print("\nCommands:")
    print("  ask <question>")
    print("  quit")
    print()

    try:
        while True:
            user_input = input("graph-rag> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                break
            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                if not question:
                    print("Please enter a question.")
                    continue
                try:
                    retrieve_and_answer(driver, qdrant, embedding_model, groq_client, question)
                except Exception as e:
                    print(f"\nERROR: {e}")
            else:
                print("Use: ask <question>")
    finally:
        driver.close()

    print("\nGraph RAG stopped.")


if __name__ == "__main__":
    main()
