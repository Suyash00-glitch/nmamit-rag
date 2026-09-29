import os
import json

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "nmamit_chunks"
CHUNKS_FILE = "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NMAMIT QDRANT VECTOR INDEX")
print("=" * 70)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")
print("Vector dimension:", model.get_sentence_embedding_dimension())


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print("\nConnecting to Qdrant Cloud...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Qdrant connection successful.")


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("\nChunks found:", len(chunks))


# ============================================================
# CREATE COLLECTION
# ============================================================

print("\nChecking collection...")

if client.collection_exists(COLLECTION_NAME):

    print(f"Collection '{COLLECTION_NAME}' already exists.")

    print("Deleting old collection...")

    client.delete_collection(COLLECTION_NAME)

    print("Old collection deleted.")


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")


# ============================================================
# PREPARE TEXT
# ============================================================

texts = []
valid_chunks = []

for chunk in chunks:

    content = chunk.get("content", "").strip()

    if not content:
        continue

    texts.append(content)
    valid_chunks.append(chunk)


print("\nValid chunks:", len(valid_chunks))


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings generated.")


# ============================================================
# UPLOAD TO QDRANT
# ============================================================

print("\nUploading vectors to Qdrant...")

points = []

for i, (chunk, embedding) in enumerate(
    zip(valid_chunks, embeddings)
):

    payload = {
        "chunk_id": chunk.get("chunk_id"),
        "content": chunk.get("content"),
        "content_type": chunk.get("content_type"),
        "source_url": chunk.get("source_url"),
        "page_title": chunk.get("page_title"),
        "section": chunk.get("section"),
        "department": chunk.get("department")
    }

    points.append(
        models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=payload
        )
    )

    # Upload in batches
    if len(points) >= 100:

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(
            f"Uploaded {i + 1}/{len(valid_chunks)}"
        )

        points = []


# ============================================================
# REMAINING POINTS
# ============================================================

if points:

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(
        f"Uploaded {len(valid_chunks)}/{len(valid_chunks)}"
    )


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING QDRANT")
print("=" * 70)

info = client.get_collection(COLLECTION_NAME)

print("Collection :", COLLECTION_NAME)
print("Points     :", info.points_count)
print("Status     :", info.status)


print("\n" + "=" * 70)
print("QDRANT INDEXING COMPLETE")
print("=" * 70)