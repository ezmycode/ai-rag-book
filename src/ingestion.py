import yaml
import os
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from src.database import init_sqlite, get_qdrant_db
from langchain_core.documents import Document

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

llm_3b = OllamaLLM(model=config["models"]["llm_model"], base_url=config["models"]["base_url"])
llm_7b = OllamaLLM(model=config["models"]["llm_model"], base_url=config["models"]["base_url"])

def extract_book_metadata(docs):
    intro_text = "\n".join([doc.page_content for doc in docs[:5]])
    prompt = f"Extract Book Name, Author, Main Characters, and a brief summary from this text. Return ONLY a pipe-separated string: Title|Author|Characters|Summary.\n\nText: {intro_text}"
    response = llm_7b.invoke(prompt)
    try:
        title, author, chars, summary = response.split("|")
        return title.strip(), author.strip(), chars.strip(), summary.strip()
    except:
        return "Unknown", "Unknown", "Unknown", "Summary unavailable"

def process_and_ingest():
    conn = init_sqlite()
    cursor = conn.cursor()
    qdrant = get_qdrant_db()
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    data_dir = config["paths"]["data_dir"]
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    total_books = len(pdf_files)
    
    print(f"\n=== Starting Ingestion: {total_books} Book(s) Found ===")

    for b_idx, file in enumerate(pdf_files, 1):
        print(f"\n[Book {b_idx}/{total_books}] Processing: {file}")
        
        loader = PyPDFLoader(os.path.join(data_dir, file))
        raw_docs = loader.load()
        
        # Extract & Save Metadata
        b_name, author, chars, b_summary = extract_book_metadata(raw_docs)
        cursor.execute("INSERT INTO books (book_name, author, characters, book_summary) VALUES (?, ?, ?, ?)", 
                       (b_name, author, chars, b_summary))
        conn.commit()
        print(f"  -> Extracted Metadata: '{b_name}' by {author}")
        
        parents = parent_splitter.split_documents(raw_docs)
        vector_docs = []
        
        for p_idx, parent in enumerate(parents):
            page_num = parent.metadata.get("page", p_idx)
            print(f"  -> Processing Page {page_num}...")
            
            p_summary = llm_3b.invoke(f"Summarize this chapter/page briefly: {parent.page_content}")
            vector_docs.append(Document(
                page_content=p_summary,
                metadata={"type": "chapter_summary", "book": b_name, "author": author, "page": page_num}
            ))
            
            children = child_splitter.split_documents([parent])
            for c_idx, child in enumerate(children):
                contextual_text = f"[{b_name} by {author}, Page {page_num}] {child.page_content}"
                vector_docs.append(Document(
                    page_content=contextual_text,
                    metadata={"type": "chunk", "book": b_name, "author": author, "page": page_num, "chunk_idx": c_idx}
                ))
        
        qdrant.add_documents(vector_docs)
        print(f"[Book {b_idx}/{total_books}] '{b_name}' successfully loaded into SQLite and Vector DB.")
        
    print("\n=== Ingestion Complete ===")

if __name__ == "__main__":
    process_and_ingest()