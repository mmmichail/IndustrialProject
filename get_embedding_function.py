from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    # Load the Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
