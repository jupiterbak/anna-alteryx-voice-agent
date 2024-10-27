import asyncio
import json
import os
from typing import List, Dict
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import faiss
from tqdm import tqdm

# Constants
OPENAI_MODEL = "text-embedding-3-small"
JSON_FILE = os.path.join("rag", "raw_data", "alteryx_docs.json")
CHROMA_COLLECTION_NAME = "alteryx_docs"
CHROMA_PERSIST_DIR = os.path.join("rag", "vectorDB", "chromaDB")
FAISS_PERSIST_DIR = os.path.join("rag", "vectorDB", "faiss")
VECTOR_DB_TO_USE = "chroma"  # or "faiss", "chroma"
EMBEDDINGS_DIMENSION = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

async def process_documents(data: Dict[str, Dict], splitter: SentenceSplitter) -> List[Document]:
    documents = []
    for item, content in tqdm(data.items(), desc="Processing documents"):
        if text := content.get("text"):
            doc = Document(text=text, metadata={"url": item, "alteryx_id": item})
            nodes = splitter.get_nodes_from_documents([doc])
            documents.extend(nodes)
    return documents

def get_vector_store(vector_db: str):
    if vector_db == "chroma":
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        return ChromaVectorStore(chroma_collection=chroma_collection)
    elif vector_db == "faiss":
        faiss_index = faiss.IndexFlatL2(EMBEDDINGS_DIMENSION)
        return FaissVectorStore(faiss_index=faiss_index)
    else:
        raise ValueError(f"Unsupported vector store: {vector_db}")

async def main() -> None:
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    vector_store = get_vector_store(VECTOR_DB_TO_USE)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=OPENAI_MODEL, dimensions=EMBEDDINGS_DIMENSION)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    nodes = await process_documents(data, splitter)

    index = VectorStoreIndex(
        nodes,
        store_nodes_override=True,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    index.storage_context.persist(persist_dir=FAISS_PERSIST_DIR if VECTOR_DB_TO_USE == "faiss" else CHROMA_PERSIST_DIR)
    
    print(f"Vector database created and saved using {VECTOR_DB_TO_USE}")
    print(f"{'ChromaDB persistence directory' if VECTOR_DB_TO_USE == 'chroma' else 'FAISS persistence directory'}: "
          f"{CHROMA_PERSIST_DIR if VECTOR_DB_TO_USE == 'chroma' else FAISS_PERSIST_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
