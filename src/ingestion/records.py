
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence


@dataclass
class ChunkRecord:

    text: str
    chunk_id: str
    document_name: str
    section_title: str
    page_number: int | None = None
    source_url: str | None = None
    header_path: list[str] = field(default_factory=list)
    score: float | None = None

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("score", None)
        return d

    @property
    def page_content(self) -> str:
        return self.text

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "source_url": self.source_url,
            "score": self.score,
        }

    def citation_label(self) -> str:
        parts = [self.document_name]
        if self.section_title:
            parts.append(f"§ {self.section_title}")
        parts.append(f"p. {self.page_number}" if self.page_number is not None else "page n/a")
        parts.append(f"[{self.chunk_id}]")
        return " — ".join(parts)


_HEADER_KEYS = ("Header 1", "Header 2", "Header 3")


def derive_section(meta: dict[str, Any]) -> tuple[str, list[str]]:
    path = [str(meta[k]).strip() for k in _HEADER_KEYS if meta.get(k)]
    section_title = path[-1] if path else "Untitled section"
    return section_title, path


# ---------------------------------------------------------------------------
# Page numbers
# ---------------------------------------------------------------------------
# cleaner.py turns <page_number>N</page_number> into [[PAGE:N]] and leaves it in
# the text, so it survives chunking. The tag sits at the END of a page, so text
# appearing BEFORE a [[PAGE:N]] belongs to page N.
#
# For each chunk, in document order:
#   - if the chunk contains sentinels, its content starts on the first one's page
#   - if it contains none, it lies entirely after the previous sentinel, so it is
#     on the page after the last one we saw
# Roman-numeral front matter (i, ii, iii...) is stripped but not recorded.

_PAGE_SENTINEL = re.compile(r"\[\[PAGE:([^\]]*)\]\]")


def strip_page_sentinels(text: str) -> tuple[str, list[int]]:
    """Return the text without sentinels, plus the numeric pages found, in order."""
    pages = [
        int(m.group(1).strip())
        for m in _PAGE_SENTINEL.finditer(text)
        if m.group(1).strip().isdigit()
    ]
    cleaned = _PAGE_SENTINEL.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), pages


# Kept so older code that imported these keeps working.
def build_page_map(markdown_text: str) -> list[tuple[int, int]]:
    return [
        (m.start(), int(m.group(1)))
        for m in re.finditer(r"<page_number>\s*(\d+)\s*</page_number>", markdown_text)
    ]


def page_for_offset(page_map: Sequence[tuple[int, int]], offset: int) -> int | None:
    if not page_map:
        return None
    current = None
    for pos, page in page_map:
        if pos <= offset:
            current = page
        else:
            break
    return current


def make_chunk_id(document_name: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_name}:{text}".encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", document_name.lower()).strip("-")[:24]
    return f"{slug}-{index:04d}-{digest}"


def build_chunk_records(
    chunks: Sequence[Any],
    document_name: str,
    source_url: str | None = None,
    raw_markdown: str | None = None,
    min_chars: int = 120,
) -> list[ChunkRecord]:

    records: list[ChunkRecord] = []
    next_page: int | None = None

    for i, ch in enumerate(chunks):
        original = getattr(ch, "page_content", None) or ""
        text, pages_here = strip_page_sentinels(original)

        if pages_here:
            page_number = pages_here[0]
            carry = pages_here[-1] + 1
        else:
            page_number = next_page
            carry = next_page

        # Advance the running page even for chunks we discard, so short chunks
        # do not desynchronise the numbering for everything after them.
        next_page = carry

        if len(text) < min_chars:
            continue

        meta = getattr(ch, "metadata", {}) or {}
        section_title, header_path = derive_section(meta)

        records.append(
            ChunkRecord(
                text=text,
                chunk_id=make_chunk_id(document_name, i, text),
                document_name=document_name,
                section_title=section_title,
                page_number=page_number,
                source_url=source_url,
                header_path=header_path,
            )
        )

    return records


def record_from_payload(payload: dict[str, Any], score: float | None = None) -> ChunkRecord:
    return ChunkRecord(
        text=payload.get("text", ""),
        chunk_id=payload.get("chunk_id", "unknown"),
        document_name=payload.get("document_name", "unknown"),
        section_title=payload.get("section_title", ""),
        page_number=payload.get("page_number"),
        source_url=payload.get("source_url"),
        header_path=payload.get("header_path", []) or [],
        score=score,
    )


if __name__ == "__main__":
    class _FakeDoc:
        def __init__(self, c, m):
            self.page_content, self.metadata = c, m

    fake = [
        _FakeDoc(
            "Chikungunya typically presents with abrupt onset of fever and severe "
            "polyarthralgia, which may persist for weeks to months after the acute "
            "phase of the illness has resolved.[[PAGE:11]]",
            {"Header 1": "2 Introduction", "Header 2": "2.3 Chikungunya"},
        ),
        _FakeDoc(
            "Zika virus infection during pregnancy is associated with congenital "
            "abnormalities, and laboratory confirmation is required to distinguish it "
            "from other arboviral infections presenting with rash.",
            {"Header 1": "2 Introduction", "Header 2": "2.4 Zika"},
        ),
        _FakeDoc("too short", {"Header 1": "2 Introduction"}),
    ]
    recs = build_chunk_records(
        fake,
        document_name="WHO Arboviral Guidelines 2025",
        source_url="https://iris.who.int/",
    )
    for r in recs:
        print(r.citation_label())
    print(f"\n{len(fake)} chunks in -> {len(recs)} records out (short chunk dropped)")
    print("Expected: first record p. 11, second record p. 12 (carried forward).")
