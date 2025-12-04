import os
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
import torch
import json
from pathlib import Path

# For robust HTML parsing
from unstructured.partition.html import partition_html

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector DB
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# LLM
from langchain_ollama import OllamaLLM as Ollama


# -------------------------------
# CONFIGURATION
# -------------------------------
RAG_INDEX_DIR = "rag_index"
FILINGS_DIR = "filings"
MAX_CONTEXT_CHARS = 8000  # Limit context size
os.makedirs(RAG_INDEX_DIR, exist_ok=True)


# -------------------------------
# EMBEDDING WRAPPER
# -------------------------------

class SentenceTransformerEmbeddings(Embeddings):
    """Custom wrapper for SentenceTransformer with dynamic device selection."""
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        if torch.cuda.is_available():
            device = "cuda"
            print("INFO: Initializing SentenceTransformer on GPU (CUDA).")
        else:
            device = "cpu"
            print("INFO: Initializing SentenceTransformer on CPU (CUDA not found).")
            
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# -------------------------------
# LANGGRAPH STATE
# -------------------------------

class RAGState(TypedDict, total=False):
    question: str
    ticker: str
    raw_text: str 
    documents: List[Document]
    retrieved_docs: List[Document]
    answer: str
    skip_processing: bool  # Flag to skip loading/chunking


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def get_index_paths(ticker: str) -> tuple:
    """Get paths for storing FAISS index and documents."""
    ticker = ticker.upper()
    faiss_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_faiss")
    docs_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_docs.json")
    metadata_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_metadata.json")
    return faiss_path, docs_path, metadata_path


def index_exists(ticker: str) -> bool:
    """Check if index files exist for a ticker."""
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    return (
        os.path.exists(faiss_path) and 
        os.path.exists(docs_path) and
        os.path.exists(metadata_path)
    )


def save_indexes(ticker: str, faiss_db: FAISS, documents: List[Document]):
    """Save FAISS index and documents as JSON."""
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    
    # Save FAISS
    faiss_db.save_local(faiss_path)
    print(f"✓ Saved FAISS index: {faiss_path}")
    
    # Save documents as JSON
    docs_json = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(docs_json, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved documents: {docs_path}")
    
    # Save metadata
    metadata = {
        "ticker": ticker,
        "num_documents": len(documents),
        "created_at": str(Path(docs_path).stat().st_mtime)
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")


def load_indexes(ticker: str, embedding_model: Embeddings) -> tuple:
    """Load FAISS and recreate BM25 from saved documents."""
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    
    # Load FAISS
    faiss_db = FAISS.load_local(
        faiss_path, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded FAISS index: {faiss_path}")
    
    # Load documents from JSON
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_json = json.load(f)
    
    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in docs_json
    ]
    print(f"✓ Loaded {len(documents)} documents from JSON")
    
    # Recreate BM25 (more reliable than pickling)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    print(f"✓ Recreated BM25 retriever from documents")
    
    return faiss_db, bm25_retriever, documents


def elements_to_text(elements):
    """Convert unstructured elements to text, preserving table structure."""
    lines = []
    for el in elements:
        text = getattr(el, "text", "").strip()
        if not text:
            continue

        if getattr(el, "category", None) == "Table":
            table_text = text.replace("\t", " | ")
            lines.append(table_text)
        else:
            lines.append(text)

    return "\n\n".join(lines)


def find_filing(ticker: str) -> Optional[Path]:
    """Find filing file for given ticker."""
    filing_dir = Path(FILINGS_DIR)
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    
    # Search patterns
    patterns = [
        f"{ticker_upper}_*.html",
        f"{ticker_upper}.html",
        f"{ticker_lower}_*.html",
        f"{ticker_lower}.html"
    ]
    
    for pattern in patterns:
        matching_files = list(filing_dir.glob(pattern))
        if matching_files:
            return matching_files[0]
    
    return None


# -------------------------------
# WORKFLOW NODES
# -------------------------------

# Node 1: Check Index
def check_index_node(state: RAGState) -> Dict[str, Any]:
    """Check if index exists and set skip flag accordingly."""
    ticker = state["ticker"].upper()
    
    if index_exists(ticker):
        print(f"📂 Index already exists for {ticker} → skipping load/chunk")
        return {"skip_processing": True}
    else:
        print(f"🔨 No index found for {ticker} → will build new index")
        return {"skip_processing": False}


# Node 2: Load Filing
def load_filing_node(state: RAGState) -> Dict[str, Any]:
    """Loads the filing HTML."""
    # Skip if we're using cached index
    if state.get("skip_processing", False):
        return {}
    
    ticker = state["ticker"].upper()
    
    filing_path = find_filing(ticker)
    if not filing_path:
        # Get available tickers for error message
        filing_dir = Path(FILINGS_DIR)
        available_files = list(filing_dir.glob("*.html"))
        available_tickers = set()
        for f in available_files:
            ticker_part = f.stem.split('_')[0].upper()
            available_tickers.add(ticker_part)
        
        raise ValueError(
            f"Filing not found for ticker {ticker}. "
            f"Available tickers: {', '.join(sorted(available_tickers)) if available_tickers else 'None'}. "
            f"Files should be in '{filing_dir.absolute()}'"
        )
    
    print(f"📄 Loading filing: {filing_path.name}")
    
    # Parse HTML using unstructured
    elements = partition_html(filename=str(filing_path))
    raw_text = elements_to_text(elements)
    
    print(f"✓ Loaded {len(raw_text):,} characters from filing")
    
    return {"raw_text": raw_text}


# Node 3: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n| ", "\n\n", "\n", " "]
)


def chunk_node(state: RAGState) -> Dict[str, Any]:
    """Splits raw text into documents."""
    # Skip if we're using cached index
    if state.get("skip_processing", False):
        return {}

    if "raw_text" not in state:
        return {}  # Defensive: should not happen

    text = state["raw_text"]
    ticker = state["ticker"]

    chunks = text_splitter.split_text(text)

    documents = [
        Document(page_content=chunk, metadata={"ticker": ticker, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    print(f"✓ Created {len(documents)} chunks")

    # Return new documents without mutating state
    return {"documents": documents}


# Node 4: Build Index and Retrieve
def build_and_retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Builds or loads the index and retrieves relevant documents."""
    question = state["question"]
    ticker = state["ticker"]
    
    # Create embedding model instance for this operation
    embedding_model = SentenceTransformerEmbeddings()
    
    # Try to load existing index
    if index_exists(ticker):
        print(f"📂 Loading existing index for {ticker}...")
        try:
            faiss_db, bm25_retriever, documents = load_indexes(ticker, embedding_model)
            print(f"✓ Index loaded successfully ({len(documents)} documents)")
        except Exception as e:
            print(f"⚠️  Failed to load index: {e}. Rebuilding...")
            if not state.get("documents"):
                return {"retrieved_docs": []}
            documents = state["documents"]
            faiss_db = FAISS.from_documents(documents, embedding_model)
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 10
            save_indexes(ticker, faiss_db, documents)
    else:
        # Build new index
        documents = state.get("documents", [])
        if not documents:
            return {"retrieved_docs": []} 

        print(f"🔨 Building new index for {ticker}...")
        
        faiss_db = FAISS.from_documents(documents, embedding_model)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10 
        
        save_indexes(ticker, faiss_db, documents)
    
    # Create retrievers
    faiss_r = faiss_db.as_retriever(search_kwargs={"k": 10})
    bm25_retriever.k = 10
    
    # Ensemble Retriever
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_r],
        weights=[0.7, 0.3] 
    )

    # Intelligent retrieval based on query type
    q_lower = question.lower()

    # Specific financial line items that need exact matching
    keyword_phrases = [
        "other operating expenses",
        "other non-current liabilities",
        "other noncurrent liabilities",
        "operating expenses",
        "cost of revenue",
        "research and development",
        "r&d",
        "sales and marketing",
        "general and administrative",
        "g&a",
    ]

    # Check if question mentions a specific line item
    phrase = next((p for p in keyword_phrases if p in q_lower), None)

    if phrase:
        print(f"✓ Keyword phrase match '{phrase}' → using hybrid retrieval")
        # Get both keyword and semantic results
        keyword_hits = [
            doc for doc in documents
            if phrase in doc.page_content.lower()
        ]
        
        semantic_docs = retriever.invoke(question)
        
        if keyword_hits:
            # Combine: prioritize exact matches but add semantic results
            seen_ids = set()
            docs = []
            
            # Add keyword matches first
            for doc in keyword_hits[:5]:
                doc_id = id(doc.page_content)
                if doc_id not in seen_ids:
                    docs.append(doc)
                    seen_ids.add(doc_id)
            
            # Add semantic matches
            for doc in semantic_docs:
                doc_id = id(doc.page_content)
                if doc_id not in seen_ids and len(docs) < 10:
                    docs.append(doc)
                    seen_ids.add(doc_id)
            
            print(f"  Found {len(keyword_hits)} exact matches, added {len(docs) - len(keyword_hits[:5])} semantic results")
        else:
            print(f"  No exact matches found, using semantic search")
            docs = semantic_docs
    else:
        # For general queries, check if asking about financial statements
        statement_keywords = ["statement of operations", "balance sheet", "income statement", 
                            "cash flow", "consolidated", "financial statements"]
        
        if any(kw in q_lower for kw in statement_keywords):
            print("✓ Financial statement query detected → boosting table chunks")
            # Get more results and prioritize chunks with table-like structure
            all_docs = retriever.invoke(question)
            
            # Prioritize chunks that look like tables
            table_docs = []
            other_docs = []
            dollar_sign = chr(36)  # Dollar sign character
            for d in all_docs:
                has_pipes = '|' in d.page_content
                has_dollars = d.page_content.count(dollar_sign) > 3
                has_numbers = sum(c.isdigit() for c in d.page_content) > 20
                if (has_pipes or has_dollars) and has_numbers:
                    table_docs.append(d)
                else:
                    other_docs.append(d)
            
            docs = table_docs[:8] + other_docs[:4]
            print(f"  Retrieved {len(table_docs[:8])} table chunks, {len(other_docs[:4])} other chunks")
        else:
            # Generic retrieval
            docs = retriever.invoke(question)[:10]
    
    print(f"✓ Retrieved {len(docs)} relevant documents")
    
    return {"retrieved_docs": docs}


# Node 5: Answer
llm = Ollama(model="llama3.2")


def answer_node(state: RAGState) -> Dict[str, Any]:
    """Generates the final answer using the LLM and retrieved context."""
    question = state["question"]
    
    if not state.get("retrieved_docs"):
        return {"answer": "I found no relevant context in the filing to answer your question."}

    # Limit context to avoid token overflow
    context_parts = []
    total_chars = 0
    
    for doc in state["retrieved_docs"]:
        content = doc.page_content
        if total_chars + len(content) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(content)
        total_chars += len(content)
    
    context = "\n\n".join(context_parts)
    print(f"ℹ️  Using {len(context_parts)} documents ({total_chars:,} chars) as context")

    # Check if this is a specific line-item query
    line_item_keywords = [
        "amount", "reported", "value", "expenses", "revenue", "income", 
        "assets", "liabilities", "equity", "how much", "what is the",
        "total", "cost of"
    ]
    
    is_line_item_query = any(keyword in question.lower() for keyword in line_item_keywords)
    
    if is_line_item_query:
        prompt = f"""You are an expert financial analyst reviewing SEC filings.

CRITICAL INSTRUCTIONS:
- Find the EXACT dollar amount or number requested
- Quote the specific line item name and value from the financial statements
- If you find multiple values (e.g., different years), report all of them with their time periods
- Use the EXACT format from the filing (e.g., "(in millions)" or "(in thousands)")
- If the exact line item is not found in the context, say so clearly

Question:
{question}

Context from SEC Filing:
{context}

Provide a precise answer with the exact amounts:"""
    else:
        prompt = f"""You are an expert financial analyst.
Use ONLY the context from the SEC filing. If the answer is not in the context, state you cannot find it.

Question:
{question}

Context:
{context}

Answer clearly and concisely:"""

    answer = llm.invoke(prompt)

    return {"answer": answer}


# -------------------------------
# BUILD WORKFLOW
# -------------------------------

workflow = StateGraph(RAGState)

workflow.add_node("check_index", check_index_node)
workflow.add_node("load_filing", load_filing_node)
workflow.add_node("chunk", chunk_node)
workflow.add_node("build_and_retrieve", build_and_retrieve_node) 
workflow.add_node("answer", answer_node)

workflow.set_entry_point("check_index")

workflow.add_edge("check_index", "load_filing")
workflow.add_edge("load_filing", "chunk")
workflow.add_edge("chunk", "build_and_retrieve")
workflow.add_edge("build_and_retrieve", "answer")
workflow.add_edge("answer", END)

rag_graph = workflow.compile()


# -------------------------------
# RAG FUNCTION
# -------------------------------

def run_rag_query(question: str, ticker: str) -> str:
    """Runs the RAG process by dynamically loading, indexing, and querying a filing."""
    ticker = ticker.upper() 
    
    try:
        result = rag_graph.invoke({
            "question": question, 
            "ticker": ticker
        })
        return result["answer"]
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        print(f"ERROR during RAG execution: {e}")
        import traceback
        traceback.print_exc()
        return "An unexpected error occurred during the financial analysis lookup."


# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def clear_index(ticker: str):
    """Clear cached index for a ticker."""
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    
    for path in [faiss_path, docs_path, metadata_path]:
        if os.path.exists(path):
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    print(f"✓ Cleared index for {ticker}")


def clear_all_indexes():
    """Clear all cached indexes."""
    import shutil
    if os.path.exists(RAG_INDEX_DIR):
        shutil.rmtree(RAG_INDEX_DIR)
        os.makedirs(RAG_INDEX_DIR)
    print("✓ Cleared all indexes")