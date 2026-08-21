from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def section_aware_chunking(
    markdown_text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits markdown text into structured chunks, preserving header hierarchy in metadata.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(md_header_splits)