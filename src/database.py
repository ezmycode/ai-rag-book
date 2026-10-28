import os
import sqlite3
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        location = config["qdrant"]["location"]
        if location != ":memory:":
            os.makedirs(location, exist_ok=True)
            _qdrant_client = QdrantClient(path=location)
        else:
            _qdrant_client = QdrantClient(location=":memory:")
    return _qdrant_client

def init_sqlite():
    db_dir = "db"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    conn = sqlite3.connect(f"{db_dir}/book_catalog.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_name TEXT,
            author TEXT,
            characters TEXT,
            book_summary TEXT
        )
    """)
    conn.commit()
    return conn

def get_qdrant_db():
    embeddings = OllamaEmbeddings(
        model=config["models"]["embedding_model"],
        base_url=config["models"]["base_url"]
    )
    client = get_qdrant_client()
    collection_name = config["qdrant"]["collection_name"]
    
    # Create the Qdrant collection if it doesn't exist yet
    if not client.collection_exists(collection_name=collection_name):
        # Obtain vector dimension from the embedding model dynamically
        sample_embedding = embeddings.embed_query("initialization_test")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(sample_embedding), 
                distance=Distance.COSINE
            )
        )
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

def is_db_ready():
    """Checks if SQLite catalog and Qdrant vector collection exist and contain data."""
    # Check SQLite
    db_path = "db/book_catalog.db"
    if not os.path.exists(db_path):
        return False, 0
    
    try:
        conn = init_sqlite()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM books")
        book_count = cursor.fetchone()[0]
        if book_count == 0:
            return False, 0
    except Exception:
        return False, 0

    # Check Qdrant Vector DB
    try:
        client = get_qdrant_client()
        collection_name = config["qdrant"]["collection_name"]
        
        if not client.collection_exists(collection_name=collection_name):
            return False, 0
            
        info = client.get_collection(collection_name=collection_name)
        if info.points_count > 0:
            return True, book_count
    except Exception:
        return False, 0

    return False, 0