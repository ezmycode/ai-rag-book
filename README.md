# Book RAG System

A Retrieval-Augmented Generation (RAG) application for querying English PDF books using **Qwen2.5:7B, Ollama, Qdrant, SQLite, and Streamlit**.

The system combines book metadata, hierarchical summaries, semantic search, and cross-encoder reranking to answer questions about books with relevant source citations.

## Features

- Query local English PDF books using natural language.
- Retrieve book titles, authors, main characters, and summaries.
- Search text chunks and page or chapter summaries.
- Route queries using book catalog metadata.
- Rerank retrieved passages for improved relevance.
- Generate answers with book, chapter, and page citations.
- Run inference locally using a single LLM: Qwen2.5:7B.

## Screenshots

### Author and Character Query

\!Author and Character Query

### Book Summary Query

\!Book Summary Query

## Architecture

The system uses SQLite for structured metadata and Qdrant for vector search.

mermaid

```
flowchart TD
    A[PDF Books] --> B[Text Extraction]
    B --> C[Metadata and Summary Extraction]
    B --> D[Chunking and Hierarchical Summaries]
    C --> E[SQLite Catalog]
    D --> F[Embeddings]
    F --> G[Qdrant Vector Store]
    H[User Query] --> I[Qwen2.5:7B Query Router]
    E --> I
    I --> J{Query Type}
    J -->|Metadata| E
    J -->|Semantic Search| G
    G --> K[Cross-Encoder Reranking]
    K --> L[Qwen2.5:7B Answer Generation]
    E --> L
    L --> M[Answer with Citations]
```

### Technology Stack

| Component | Purpose |
| --- | --- |
| Python | Application logic |
| Streamlit | Web interface |
| Ollama | Local LLM inference |
| Qwen2.5:7B | Metadata extraction, summarization, query routing, and answer generation |
| SQLite | Book catalog and structured metadata |
| Qdrant | Vector storage and semantic retrieval |
| Embedding model | Text embeddings for semantic search |
| Cross-encoder | Reranking retrieved passages |

**LLM requirement:** Qwen2.5:7B is the only language model used for LLM tasks. Embedding models and cross-encoders are separate components and may use their own model weights.

## Prerequisites

- Python 3.12
- [uv](<https://docs.astral.sh/uv/>)
- [Ollama](<https://ollama.com/>)
- [Qdrant](<https://qdrant.tech/>)
- English PDF books
- The project source code and `requirements.txt`

## Installation

### 1\. Clone the Repository

bash

```
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2\. Install Qwen2.5:7B

Install Ollama using its [official installation instructions](<https://ollama.com/>).

Then run on macOS, Windows, or Linux:

bash

```
ollama pull qwen2.5:7b
```

### 3\. Create a Python Virtual Environment

**macOS and Linux**

bash

```
uv venv .venv --python 3.12
source .venv/bin/activate
```

**Windows PowerShell**

powershell

```
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
```

### 4\. Install Dependencies

Run on all platforms with the virtual environment activated:

bash

```
uv pip install -r requirements.txt
```

### 5\. Configure Qdrant

Ensure Qdrant is running and configure its connection details, collection settings, embedding model, and SQLite database path according to your application.

Refer to the [Qdrant documentation](<https://qdrant.tech/documentation/>) for setup options.

### 6\. Prepare PDF Books

Place your English PDF files in the `data/` directory.

Example:

text

```
data/
├── The_Great_Gatsby.pdf
├── Pride_and_Prejudice.pdf
└── Moby_Dick.pdf
```

Use descriptive filenames to make book identification easier.

## Run the Application

Start the Streamlit interface:

bash

```
streamlit run app.py
```

Then:

1. Open the local URL displayed in the terminal.
2. Click **Load PDFs to Database** in the sidebar.
3. Wait for ingestion and indexing to finish.
4. Enter your question in the main interface.
5. Review the answer and available citations.

## Example Queries

- Who wrote _The Great Gatsby_?
- Summarize _Pride and Prejudice_.
- Who are the main characters in this book?
- What motivates the protagonist?
- Summarize a specific chapter.
- Find passages discussing a particular theme.

## Project Structure

text

```
book-rag-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── docs/
    └── screenshots/
        ├── Author Character Query.png
        └── Book Summary Query.png
```

This is an illustrative structure; your actual project may contain additional modules and configuration files.

## Notes

- Keep `.venv/`, secrets, and private configuration files out of Git.
- Add an `.env.example` file if your application requires environment variables.
- Configure Qdrant persistence if you need indexed data to survive restarts.
- PDF extraction quality affects retrieval and citation accuracy.
- Model performance depends on available hardware and configuration.
- Only distribute PDF books for which you have the necessary rights.

## License

Add a `LICENSE` file specifying the terms under which others may use and contribute to this project.

 