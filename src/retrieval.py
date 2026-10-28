import json
import sqlite3
import yaml
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM
from src.database import init_sqlite, get_qdrant_db

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

llm_7b = OllamaLLM(model=config["models"]["llm_model"], base_url=config["models"]["base_url"])
reranker = CrossEncoder(config["models"]["reranker_model"])

def get_catalog_context():
    conn = init_sqlite()
    cursor = conn.cursor()
    cursor.execute("SELECT book_name, author FROM books")
    catalog = cursor.fetchall()
    return [f"{b[0]} by {b[1]}" for b in catalog]

def query_router(user_query, catalog):
    prompt = f"""
    Available Books: {catalog}
    User Query: "{user_query}"
    
    Determine the search parameters. Output ONLY a valid JSON object with these keys:
    - target_book: (string or null)
    - target_author: (string or null)
    - search_type: ("metadata_only" if query is just asking for author/character names/summaries, else "vector_search")
    """
    response = llm_7b.invoke(prompt)
    try:
        return json.loads(response)
    except:
        return {"target_book": None, "target_author": None, "search_type": "vector_search"}

def retrieve_and_rerank(query):
    catalog = get_catalog_context()
    route_params = query_router(query, catalog)
    
    # 1. Metadata Only Route (SQL)
    if route_params.get("search_type") == "metadata_only":
        conn = init_sqlite()
        cur = conn.cursor()
        book = route_params.get("target_book") or ""
        cur.execute("SELECT book_name, author, characters, book_summary FROM books WHERE book_name LIKE ?", (f"%{book}%",))
        res = cur.fetchone()
        if res:
            answer = f"### 📖 {res[0]}\n**Author:** {res[1]}\n**Key Characters:** {res[2]}\n\n**Book Summary:**\n{res[3]}"
            return answer, [] # Empty list because there are no vector chunks to cite
        
    # 2. Vector Search Route (Qdrant)
    qdrant = get_qdrant_db()
    filters = {}
    if route_params.get("target_book"):
        filters["book"] = route_params["target_book"]
        
    retriever = qdrant.as_retriever(search_kwargs={"k": 20, "filter": filters if filters else None})
    initial_docs = retriever.invoke(query)
    
    if not initial_docs:
        return "No relevant information found in the library.", []
    
    # Reranking
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)[:5]
    final_docs = [doc for score, doc in scored_docs]
    
    # Final LLM Synthesis
    context = "\n".join([f"(Book: {d.metadata.get('book')} | Page: {d.metadata.get('page')}): {d.page_content}" for d in final_docs])
    synthesis_prompt = f"Using this context, answer the query. Include citations referencing the Book Name and Page.\nContext:\n{context}\n\nQuery: {query}"
    
    answer = llm_7b.invoke(synthesis_prompt)
    return answer, final_docs