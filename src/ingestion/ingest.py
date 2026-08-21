
from __future__ import annotations

import uuid
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from config import COLLECTION_NAME, QDRANT_PATH
from ingestion.embedder import create_embeddings
from ingestion.records import ChunkRecord


_NAMESPACE = uuid.UUID("6f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def run_ingestion(
    records: Sequence[ChunkRecord],
    batch_size: int = 64,
    recreate: bool = True,
) -> int:
    """
    Embed and upsert records.

    recreate=True drops the collection first. chunk_ids are content hashes, so
    re-ingesting after a text change produces NEW ids — without dropping, the old
    points would linger and retrieval would mix stale chunks (with no page
    numbers) into the results.
    """
    records = [r for r in records if r.text.strip()]
    if not records:
        print("[Ingest] No records to ingest.")
        return 0

    print(f"[Ingest] Embedding {len(records)} chunks...")
    vectors = create_embeddings([r.text for r in records])
    vector_size = len(vectors[0])

    client = QdrantClient(path=QDRANT_PATH)
    try:
        if recreate and client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print(f"[Ingest] Dropped stale collection '{COLLECTION_NAME}'.")

        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"[Ingest] Created collection '{COLLECTION_NAME}' (dim={vector_size}).")

        points = [
            PointStruct(
                id=_point_id(rec.chunk_id),
                vector=vectors[i],
                payload=rec.to_payload(),
            )
            for i, rec in enumerate(records)
        ]

        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[start : start + batch_size],
                wait=True,
            )

        total = client.count(COLLECTION_NAME, exact=True).count
        with_pages = sum(1 for r in records if r.page_number is not None)
        print(f"[Ingest] Upserted {len(points)} chunks. Collection total: {total}.")
        print(f"[Ingest] Page numbers present on {with_pages}/{len(records)} chunks.")
        return len(points)
    finally:
        client.close()
