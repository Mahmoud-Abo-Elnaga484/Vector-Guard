

from __future__ import annotations

from typing import Sequence

import streamlit as st
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH
from ingestion.records import ChunkRecord, record_from_payload

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@st.cache_resource
def _get_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def _get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def embed_query(query: str) -> list[float]:
    text = query if query.startswith(BGE_QUERY_PREFIX) else BGE_QUERY_PREFIX + query
    return _get_embedder().embed_query(text)


def search_documents(
    query: str,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[ChunkRecord]:

    if not query or not query.strip():
        return []

    hits = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
    ).points

    return [record_from_payload(h.payload or {}, score=h.score) for h in hits]


def format_evidence_block(records: Sequence[ChunkRecord]) -> str:

    if not records:
        return "RETRIEVED EVIDENCE: (empty — no passages passed the retrieval threshold)"

    blocks = []
    for r in records:
        header = f"[{r.chunk_id}] {r.document_name}"
        if r.section_title:
            header += f" | section: {r.section_title}"
        header += f" | page: {r.page_number if r.page_number is not None else 'not available'}"
        header += f" | score: {r.score:.3f}" if r.score is not None else ""
        blocks.append(f"{header}\n{r.text}")

    return "### RETRIEVED EVIDENCE\n\n" + "\n\n---\n\n".join(blocks)
