import streamlit as st
import yaml
from src.ingestion import process_and_ingest
from src.retrieval import retrieve_and_rerank
from src.database import is_db_ready

st.set_page_config(page_title="Library RAG", layout="wide")
st.title("📚 360° Book Query Assistant")

# Check if persisted database exists
db_exists, book_count = is_db_ready()

# Sidebar Controls
with st.sidebar:
    st.header("Admin Controls")
    st.write("Ensure your PDFs are placed in the `data/` folder.")
    
    if db_exists:
        st.success(f"🟢 Database Ready ({book_count} book(s) indexed)")
    else:
        st.warning("🔴 No persistent database found.")

    if st.button("Load / Re-index PDFs"):
        with st.spinner("Ingesting PDFs... Check terminal for page-level logs."):
            process_and_ingest()
            st.success("Database created and saved to disk!")
            st.rerun()

# Query Interface
if db_exists:
    query = st.text_input("Ask a question about your books (e.g., characters, plots, authors, summaries):")

    if query:
        with st.spinner("Analyzing query and retrieving context..."):
            response, relevant_docs = retrieve_and_rerank(query)
            
            st.markdown("### Answer")
            st.write(response)
            
            if relevant_docs:
                with st.expander("View Source References (Vector Chunks)"):
                    for i, doc in enumerate(relevant_docs):
                        book = doc.metadata.get('book', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        chunk_type = doc.metadata.get('type', 'chunk')
                        
                        st.markdown(f"**Reference {i+1}: {book} (Page {page}) - [{chunk_type}]**")
                        st.caption(doc.page_content)
else:
    st.info("👈 No database found on disk. Please place your PDF files into `data/` and click **Load / Re-index PDFs** in the sidebar to initialize persistent storage.")