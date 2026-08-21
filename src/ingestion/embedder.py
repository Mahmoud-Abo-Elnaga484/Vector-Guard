from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def create_embeddings(content_list):
    encode_kwargs = {'normalize_embeddings': True}
    
    hf_embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs=encode_kwargs
    )
    
    embeddings = hf_embedder.embed_documents(content_list)
    return embeddings