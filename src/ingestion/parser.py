import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from llama_cloud import LlamaCloud

load_dotenv(find_dotenv())

def parse_pdf_to_markdown(input_pdf_path: str | Path, output_md_path: str | Path) -> str:

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is missing in .env file")

    in_path = Path(input_pdf_path)
    out_path = Path(output_md_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = LlamaCloud(api_key=api_key)

    print(f"[Parser] Uploading {in_path.name} to LlamaCloud...")
    with open(in_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="parse")

    print("[Parser] Parsing file with 'agentic' tier...")
    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full", "text_full"],
    )

    markdown_content = result.markdown_full or ""

    out_path.write_text(markdown_content, encoding="utf-8")

    print(f"[Parser] Done! Saved Markdown to: {out_path}")
    print(f"[Parser] Total Characters: {len(markdown_content)}")

    return markdown_content


if __name__ == "__main__":
    input_file = Path("src/assets/documents/ChikungunyaFile.pdf")
    output_file = Path("src/assets/parsed/ChikungunyaFile.md")

    if input_file.exists():
        parse_pdf_to_markdown(input_file, output_file)
    else:
        print(f"Please place your PDF file in: {input_file}")