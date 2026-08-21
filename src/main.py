from __future__ import annotations

import sys
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLEANED_DATA_DIR,
    DEMO_TESTSET_PATH,
    DOCUMENT_NAME,
    DOCUMENT_NAMES,
    PARSED_DATA_DIR,
    RAW_DOCS_DIR,
    SOURCE_URL,
    TOP_K,
)


def cmd_parse() -> None:
    from ingestion.parser import parse_pdf_to_markdown

    pdfs = sorted(Path(RAW_DOCS_DIR).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {RAW_DOCS_DIR}")
    for pdf in pdfs:
        parse_pdf_to_markdown(pdf, Path(PARSED_DATA_DIR) / f"{pdf.stem}.md")


def cmd_clean() -> None:
    from ingestion.cleaner import clean_file

    parsed_files = sorted(Path(PARSED_DATA_DIR).glob("*.md"))
    if not parsed_files:
        sys.exit(f"Not found: no .md files in {PARSED_DATA_DIR}. Run `python main.py parse` first.")

    for src in parsed_files:
        clean_file(src, Path(CLEANED_DATA_DIR) / src.name)

    # Flag any source PDF that was never parsed, so it doesn't silently drop
    # out of the index the way ChikungunyaFile.pdf used to.
    pdf_stems = {p.stem for p in Path(RAW_DOCS_DIR).glob("*.pdf")}
    cleaned_stems = {p.stem for p in parsed_files}
    missing = pdf_stems - cleaned_stems
    if missing:
        print(
            f"[Main] WARNING: {sorted(missing)} have a PDF in {RAW_DOCS_DIR} but no "
            "parsed .md — they will NOT be ingested. Run `python main.py parse` first "
            "(needs LLAMA_CLOUD_API_KEY)."
        )


def cmd_ingest() -> None:
    from ingestion.chunker import section_aware_chunking
    from ingestion.ingest import run_ingestion
    from ingestion.records import build_chunk_records

    cleaned_files = sorted(Path(CLEANED_DATA_DIR).glob("*.md"))
    if not cleaned_files:
        sys.exit(f"Not found: no .md files in {CLEANED_DATA_DIR}. Run `python main.py clean` first.")

    all_records = []
    for src in cleaned_files:
        raw = src.read_text(encoding="utf-8")
        print(f"[Main] Loaded {len(raw)} chars from {src}")

        if "[[PAGE:" not in raw:
            print(
                f"[Main] WARNING: no [[PAGE:n]] sentinels in {src.name}. Re-run "
                "`python main.py clean` with the updated cleaner, or citations will "
                "show 'page n/a'."
            )

        chunks = section_aware_chunking(raw, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"[Main] Chunker produced {len(chunks)} chunks from {src.name}.")

        doc_name, source_url = DOCUMENT_NAMES.get(src.name, (src.stem, SOURCE_URL))
        records = build_chunk_records(
            chunks,
            document_name=doc_name,
            source_url=source_url,
            raw_markdown=raw,
        )
        print(f"[Main] Built {len(records)} chunk records from {src.name}.")
        all_records.extend(records)

    with_pages = sum(1 for r in all_records if r.page_number is not None)
    if with_pages == 0:
        print("[Main] WARNING: no page numbers recovered. Citations will show 'page n/a'.")
    else:
        pages = [r.page_number for r in all_records if r.page_number is not None]
        print(
            f"[Main] Page numbers recovered for {with_pages}/{len(all_records)} records "
            f"(range {min(pages)}–{max(pages)})."
        )

    print(f"[Main] Ingesting {len(all_records)} total records from {len(cleaned_files)} document(s).")
    run_ingestion(all_records)


def cmd_query(query: str) -> None:
    from generation.generator import generate_differential

    if not query:
        sys.exit('Usage: python main.py query "your clinical question"')
    print(generate_differential(query, top_k=TOP_K).to_markdown())


def cmd_sections() -> None:
    from evaluation.dump_sections import main as dump_main

    dump_main()


def cmd_eval(extra: list[str]) -> None:
    from evaluation.run_eval import main as run_eval_main

    sys.argv = ["run_eval"] + extra
    run_eval_main()


def cmd_demo() -> None:
    """The 3-case evaluation the agenda asks for."""
    from evaluation.run_eval import main as run_eval_main

    sys.argv = ["run_eval", "--testset", str(DEMO_TESTSET_PATH)]
    run_eval_main()


COMMANDS = {"parse", "clean", "ingest", "query", "eval", "demo", "sections"}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in COMMANDS:
        sys.exit(f"Usage: python main.py [{'|'.join(sorted(COMMANDS))}]")

    if cmd == "parse":
        cmd_parse()
    elif cmd == "clean":
        cmd_clean()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "query":
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "sections":
        cmd_sections()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "eval":
        cmd_eval(sys.argv[2:])
