"""
Print the section titles and page coverage actually present in the Qdrant
collection. Use this to verify the relevant_sections annotations in testset.json
before reporting Precision@k.

    python main.py sections
"""

from __future__ import annotations

from collections import Counter

from qdrant_client import QdrantClient

from config import COLLECTION_NAME, QDRANT_PATH


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' does not exist. Run: python main.py ingest")
            return

        total = client.count(COLLECTION_NAME, exact=True).count
        sections: Counter[str] = Counter()
        header_terms: Counter[str] = Counter()
        with_page = 0
        pages = []

        offset = None
        seen = 0
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                seen += 1
                sections[payload.get("section_title", "") or "(none)"] += 1
                for h in payload.get("header_path", []) or []:
                    header_terms[h] += 1
                page = payload.get("page_number")
                if page is not None:
                    with_page += 1
                    pages.append(page)
            if offset is None:
                break

        print(f"Collection : {COLLECTION_NAME}")
        print(f"Chunks     : {total} (scanned {seen})")
        if pages:
            print(f"Page numbers: {with_page}/{seen} chunks, range {min(pages)}–{max(pages)}")
        else:
            print(f"Page numbers: 0/{seen} chunks — citations will read 'page n/a'")

        print("\n--- section_title values (deepest header per chunk) ---")
        for name, count in sorted(sections.items()):
            print(f"{count:4d}  {name}")

        print("\n--- every header seen anywhere in a header_path ---")
        print("(Precision@k matches against these too, so you may annotate at any level.)")
        for name, count in sorted(header_terms.items()):
            print(f"{count:4d}  {name}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
