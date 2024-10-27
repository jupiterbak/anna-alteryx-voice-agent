import os
from typing import Literal
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexLLMRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
import faiss

# Constants
OPENAI_MODEL = "text-embedding-3-small"
JSON_FILE = os.path.join("rag", "raw_data", "alteryx_docs.json")
CHROMA_COLLECTION_NAME = "alteryx_docs"
CHROMA_PERSIST_DIR = os.path.join("rag", "vectorDB", "chromaDB")
FAISS_PERSIST_DIR = os.path.join("rag", "vectorDB", "faiss")
VECTOR_DB_TO_USE: Literal["faiss", "chroma"] = "chroma"
EMBEDDINGS_DIMENSION = 1536

# Configure LlamaIndex settings
Settings.llm = OpenAI(model='gpt-4')
Settings.embed_model = OpenAIEmbedding(model=OPENAI_MODEL)

def get_vector_store(db_type: Literal["faiss", "chroma"]):
    if db_type == "faiss":
        faiss.IndexFlatL2(EMBEDDINGS_DIMENSION)  # Initialize FAISS index
        return FaissVectorStore.from_persist_dir(FAISS_PERSIST_DIR)
    elif db_type == "chroma":
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        return ChromaVectorStore(chroma_collection=chroma_collection)
    else:
        raise ValueError(f"Unsupported vector store: {db_type}")

def setup_query_engine(vector_store):
    if VECTOR_DB_TO_USE == "faiss":
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=FAISS_PERSIST_DIR
        )
        index = load_index_from_storage(storage_context=storage_context)  
    else:
        index = VectorStoreIndex.from_vector_store(vector_store, settings=Settings)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    return RetrieverQueryEngine(retriever=retriever)

def main():
    vector_store = get_vector_store(VECTOR_DB_TO_USE)
    query_engine = setup_query_engine(vector_store)
    
    query = "How do I use the Alteryx Designer?"
    result = query_engine.retrieve(query)
    print(result)

if __name__ == "__main__":
    main()
