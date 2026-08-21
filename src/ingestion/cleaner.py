"""
Markdown cleaner.

IMPORTANT CHANGE (page numbers fix)
-----------------------------------
The previous version DELETED every <page_number>N</page_number> tag. records.py
then searched the *cleaned* text for those tags to build a page map, found none,
and every citation ended up as "page n/a".

This version converts each tag into a compact sentinel [[PAGE:N]] that survives
cleaning and chunking. records.py reads the sentinel, records the page number,
and strips the sentinel from the text before it is embedded or stored — so the
sentinel never reaches the vector index or the model.

LlamaParse emits the tag at the END of a page (it is the page footer), so the
text immediately BEFORE a [[PAGE:N]] belongs to page N. records.py relies on
that ordering.
"""

import re
from pathlib import Path

PAGE_TAG = re.compile(r"<page_number>\s*(.*?)\s*</page_number>", re.DOTALL)


def clean_markdown(text: str) -> str:
    # Page markers: convert, do NOT delete.
    text = PAGE_TAG.sub(lambda m: f"[[PAGE:{m.group(1).strip()}]]", text)

    noise_line_patterns = [
        r"^\s*WHO logo.*$",
        r"^\s*Introduction( arrows)? icon\s*$",
        r"^\s*Introduction Introduction icon\s*$",
        r"^.{0,160}\b(WHO logo|PAHO/WHO logo)(\s+\w+)?\s*$",
    ]
    for pat in noise_line_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_file(input_path: str | Path, output_path: str | Path) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    n_pages = len(PAGE_TAG.findall(raw))
    cleaned = clean_markdown(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    removed_chars = len(raw) - len(cleaned)
    print(f"[Cleaner] Done. Removed {removed_chars} characters of noise.")
    print(f"[Cleaner] Original: {len(raw)} chars -> Cleaned: {len(cleaned)} chars")
    print(f"[Cleaner] Preserved {n_pages} page markers as [[PAGE:n]] sentinels.")
    if n_pages == 0:
        print(
            "[Cleaner] WARNING: the source markdown contains no <page_number> tags. "
            "Citations will show 'page n/a'."
        )


if __name__ == "__main__":
    default_input = Path("assets/parsed/MainFile.md")
    default_output = Path("assets/cleaned/MainFile.md")

    if default_input.exists():
        clean_file(default_input, default_output)
    else:
        print(f"File not found: {default_input}")
