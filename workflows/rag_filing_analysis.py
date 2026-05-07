import os
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
import torch
import json
from pathlib import Path
import hashlib
import requests
from dotenv import load_dotenv

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

# SEC API
from sec_api import QueryApi

load_dotenv()

RAG_INDEX_DIR = "rag_index"
FILINGS_DIR = "filings"
MAX_CONTEXT_CHARS = 8000  
os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(FILINGS_DIR, exist_ok=True)


class SentenceTransformerEmbeddings(Embeddings):
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




class RAGState(TypedDict, total=False):
    question: str
    ticker: str
    raw_text: str 
    documents: List[Document]
    retrieved_docs: List[Document]
    answer: str
    skip_processing: bool  



def get_index_paths(ticker: str) -> tuple:
    ticker = ticker.upper()
    faiss_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_faiss")
    docs_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_docs.json")
    metadata_path = os.path.join(RAG_INDEX_DIR, f"{ticker}_metadata.json")
    return faiss_path, docs_path, metadata_path


def index_exists(ticker: str) -> bool:
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    return (
        os.path.exists(faiss_path) and 
        os.path.exists(docs_path) and
        os.path.exists(metadata_path)
    )


def save_indexes(ticker: str, faiss_db: FAISS, documents: List[Document]):
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    
    
    faiss_db.save_local(faiss_path)
    print(f"✓ Saved FAISS index: {faiss_path}")
    
    
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
    
 
    metadata = {
        "ticker": ticker,
        "num_documents": len(documents),
        "created_at": str(Path(docs_path).stat().st_mtime)
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")


def load_indexes(ticker: str, embedding_model: Embeddings) -> tuple:
    faiss_path, docs_path, metadata_path = get_index_paths(ticker)
    
    
    faiss_db = FAISS.load_local(
        faiss_path, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded FAISS index: {faiss_path}")
    
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_json = json.load(f)
    
    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in docs_json
    ]
    print(f"✓ Loaded {len(documents)} documents from JSON")
    
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    print(f"✓ Recreated BM25 retriever from documents")
    
    return faiss_db, bm25_retriever, documents


def elements_to_text(elements):
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
    filing_dir = Path(FILINGS_DIR)
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    
   
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

def download_sec_filing(ticker: str, filing_type: str = "10-K") -> Optional[Path]:
    ticker = ticker.upper()
    api_key = os.getenv("SEC_API_KEY")
    
    if not api_key:
        print(f"⚠️  SEC_API_KEY not found. Cannot download filing for {ticker}.")
        return None
    
    try:
        print(f"📥 Downloading latest {filing_type} filing for {ticker}...")
        
        
        query_api = QueryApi(api_key=api_key)
        
      
        query = {
            "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"{filing_type}\""}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        
        resp = query_api.get_filings(query)
        
        if not resp.get("filings"):
            print(f"❌ No {filing_type} filing found for {ticker}.")
            return None
        
        filing = resp["filings"][0]
        filing_url = filing.get("linkToFilingDetails")
        filing_date = filing.get("filedAt", "")
        filing_accession = filing.get("accessionNo", "")
        
        if not filing_url:
            print(f"❌ No filing URL found for {ticker}.")
            return None
        
        
        if filing_accession:
            file_hash = hashlib.md5(filing_accession.encode()).hexdigest()[:16]
        else:
           
            file_hash = hashlib.md5(filing_url.encode()).hexdigest()[:16]
        
       
        filename = f"{ticker}_{filing_type}_{file_hash}.html"
        file_path = Path(FILINGS_DIR) / filename
        
       
        if file_path.exists():
            print(f"✓ Filing already exists: {filename}")
            return file_path
        
        
        headers = {"User-Agent": "FinSight Agent (contact: pranaybhagwat04@gmail.com)"}
        response = requests.get(filing_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        html_content = response.text
        
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size_mb = len(html_content) / (1024 * 1024)
        print(f"✓ Successfully downloaded and saved: {filename} ({file_size_mb:.2f} MB)")
        print(f"  Filing date: {filing_date}")
        print(f"  URL: {filing_url}")
        
        return file_path
        
    except requests.RequestException as e:
        print(f"❌ Network error downloading filing for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error downloading filing for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_index_node(state: RAGState) -> Dict[str, Any]:
   
    ticker = state["ticker"].upper()
    
    if index_exists(ticker):
        print(f"📂 Index already exists for {ticker} → skipping load/chunk")
        return {"skip_processing": True}
    else:
        print(f"🔨 No index found for {ticker} → will build new index")
        return {"skip_processing": False}

def load_filing_node(state: RAGState) -> Dict[str, Any]:
    if state.get("skip_processing", False):
        return {}
    
    ticker = state["ticker"].upper()
    
    
    filing_path = find_filing(ticker)
    
    
    if not filing_path:
        print(f"📥 Filing not found locally for {ticker}. Attempting to download...")
        filing_path = download_sec_filing(ticker, filing_type="10-K")
        
       
        if not filing_path:
            print(f"📥 10-K not available. Trying 10-Q for {ticker}...")
            filing_path = download_sec_filing(ticker, filing_type="10-Q")
        
       
        if not filing_path:
            filing_dir = Path(FILINGS_DIR)
            available_files = list(filing_dir.glob("*.html"))
            available_tickers = set()
            for f in available_files:
                ticker_part = f.stem.split('_')[0].upper()
                available_tickers.add(ticker_part)
            
            raise ValueError(
                f"Could not find or download filing for ticker {ticker}. "
                f"Available tickers: {', '.join(sorted(available_tickers)) if available_tickers else 'None'}. "
                f"Please ensure SEC_API_KEY is set in environment variables."
            )
    
    print(f"📄 Loading filing: {filing_path.name}")
    
    elements = partition_html(filename=str(filing_path))
    raw_text = elements_to_text(elements)
    
    print(f"✓ Loaded {len(raw_text):,} characters from filing")
    
    return {"raw_text": raw_text}


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n| ", "\n\n", "\n", " "]
)


def chunk_node(state: RAGState) -> Dict[str, Any]:
    if state.get("skip_processing", False):
        return {}

    if "raw_text" not in state:
        return {}  

    text = state["raw_text"]
    ticker = state["ticker"]

    chunks = text_splitter.split_text(text)

    documents = [
        Document(page_content=chunk, metadata={"ticker": ticker, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    print(f"✓ Created {len(documents)} chunks")

    return {"documents": documents}


def build_and_retrieve_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    ticker = state["ticker"]
    
    
    embedding_model = SentenceTransformerEmbeddings()
    
   
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
       
        documents = state.get("documents", [])
        if not documents:
            return {"retrieved_docs": []} 

        print(f"🔨 Building new index for {ticker}...")
        
        faiss_db = FAISS.from_documents(documents, embedding_model)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10 
        
        save_indexes(ticker, faiss_db, documents)
    
    
    faiss_r = faiss_db.as_retriever(search_kwargs={"k": 10})
    bm25_retriever.k = 10
    
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_r],
        weights=[0.7, 0.3] 
    )

    q_lower = question.lower()

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

    
    phrase = next((p for p in keyword_phrases if p in q_lower), None)

    if phrase:
        print(f"✓ Keyword phrase match '{phrase}' → using hybrid retrieval")
        
        keyword_hits = [
            doc for doc in documents
            if phrase in doc.page_content.lower()
        ]
        
        semantic_docs = retriever.invoke(question)
        
        if keyword_hits:
           
            seen_ids = set()
            docs = []
            
           
            for doc in keyword_hits[:5]:
                doc_id = id(doc.page_content)
                if doc_id not in seen_ids:
                    docs.append(doc)
                    seen_ids.add(doc_id)
      
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
       
        statement_keywords = ["statement of operations", "balance sheet", "income statement", 
                            "cash flow", "consolidated", "financial statements"]
        
        if any(kw in q_lower for kw in statement_keywords):
            print("✓ Financial statement query detected → boosting table chunks")
           
            all_docs = retriever.invoke(question)
            
            
            table_docs = []
            other_docs = []
            dollar_sign = chr(36) 
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
         
            docs = retriever.invoke(question)[:10]
    
    print(f"✓ Retrieved {len(docs)} relevant documents")
    
    return {"retrieved_docs": docs}



llm = Ollama(model="llama3.2")


def answer_node(state: RAGState) -> Dict[str, Any]:
   
    question = state["question"]
    
    if not state.get("retrieved_docs"):
        return {"answer": "I found no relevant context in the filing to answer your question."}

   
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

def run_rag_query(question: str, ticker: str) -> str:
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


def clear_index(ticker: str):
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
    import shutil
    if os.path.exists(RAG_INDEX_DIR):
        shutil.rmtree(RAG_INDEX_DIR)
        os.makedirs(RAG_INDEX_DIR)
    print("✓ Cleared all indexes")


def download_filing_for_ticker(ticker: str, filing_type: str = "10-K") -> Optional[Path]:
    return download_sec_filing(ticker, filing_type)


def list_available_filings() -> List[str]:
    filing_dir = Path(FILINGS_DIR)
    if not filing_dir.exists():
        return []
    
    available_files = list(filing_dir.glob("*.html"))
    available_tickers = set()
    
    for f in available_files:
        ticker_part = f.stem.split('_')[0].upper()
        available_tickers.add(ticker_part)
    
    return sorted(list(available_tickers))