
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List,Tuple
import json
import uuid
import os
import base64
from datetime import datetime,timedelta
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import asyncio
from pathlib import Path
from database import db, init_db, close_db
from contextlib import asynccontextmanager
import re
from state import FinanceAgentState
from workflows.extract_entities import extract_entities_node
from workflows.get_stock_data_and_chart import get_stock_data_and_chart_node
from workflows.get_financial_news import get_financial_news_node
from workflows.get_sec_filing_section import get_sec_filing_section_node
from workflows.curate_report import curate_report_node
from workflows.rag_filing_analysis import run_rag_query
from langchain_ollama import OllamaLLM
from fastapi.staticfiles import StaticFiles
from bson import ObjectId
from fastapi.responses import Response,FileResponse
import mimetypes
from authentication import (
    UserRegister, UserLogin, Token,
    create_user, authenticate_user, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up FinSight API...")
    try:
        await init_db()
        print("✓ Startup complete")
    except Exception as e:
        print(f"✗ Startup failed: {e}")
    yield 
    print("Shutting down FinSight API...")
    await close_db()
    print("✓ Shutdown complete")

app = FastAPI(title="FinSight API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("static", exist_ok=True)


app.mount("/charts", StaticFiles(directory="charts"), name="charts")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")



@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("static/frontend.html")

# Initialize LLM
try:
    llm = OllamaLLM(model="llama3.2", format="json")
    print("✓ Ollama LLM initialized")
except Exception as e:
    print(f"✗ Error initializing Ollama: {e}")
    llm = None

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


def detect_follow_up(state: FinanceAgentState) -> bool:
    query_lower = state["user_query"].lower()
    
    follow_up_indicators = [
        # Continuation words
        "also", "and", "now", "then", "next",
        # Comparative words
        "what about", "how about", "versus", "vs", "compared to", "compare",
        # Additive words
        "too", "as well", "additionally",
        # Reference words
        "same for", "do the same", "repeat for"
    ]
    
    return any(indicator in query_lower for indicator in follow_up_indicators)


def resolve_coreferences(query: str, state: FinanceAgentState) -> str:
    resolved_query = query
    last_company = state.get("company_name")
    last_ticker = state.get("ticker")
    
    if not last_company:
        return query  
    
    possessive_pronouns = {
        r'\bits\b': f"{last_company}'s",
        r'\btheir\b': f"{last_company}'s",
        r'\bits\'\b': f"{last_company}'s",
    }
    
    subject_pronouns = {
        r'\bit\b': last_company,
        r'\bthey\b': last_company,
        r'\bthem\b': last_company,
    }
    
    references = {
        r'\bthat company\b': last_company,
        r'\bthis company\b': last_company,
        r'\bthe company\b': last_company,
        r'\bsame company\b': last_company,
    }

    for pattern, replacement in {**possessive_pronouns, **subject_pronouns, **references}.items():
        resolved_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)
    
    print(f"🔄 Resolved: '{query}' -> '{resolved_query}'")
    return resolved_query


def extract_entities_with_context(state: FinanceAgentState) -> dict:
    entities = extract_entities_node(state)
    
    is_follow_up = detect_follow_up(state)
    
    if is_follow_up:
        if not entities.get("ticker") and state.get("ticker"):
            print(f"💡 Follow-up detected - using previous ticker: {state.get('ticker')}")
            entities["ticker"] = state.get("ticker")
            entities["company_name"] = state.get("company_name")
        
        if not entities.get("filing_type") and state.get("filing_type"):
            entities["filing_type"] = state.get("filing_type")
    
    query_lower = state["user_query"].lower()
    has_pronoun = any(word in query_lower for word in ["it", "its", "they", "their", "them"])
    has_reference = any(phrase in query_lower for phrase in ["that company", "this company", "the company"])
    
    if (has_pronoun or has_reference) and not entities.get("ticker") and state.get("ticker"):
        print(f"💡 Pronoun/reference detected - using previous ticker: {state.get('ticker')}")
        entities["ticker"] = state.get("ticker")
        entities["company_name"] = state.get("company_name")
    
    return entities


def analyze_conversation_context(state: FinanceAgentState) -> dict:
    current_query = state["user_query"].lower()
    messages = state.get("messages", [])
    
    context = {
        "is_continuation": False,
        "is_comparison": False,
        "is_new_topic": True,
        "previous_intent": state.get("intent"),
        "previous_company": state.get("company_name"),
        "previous_ticker": state.get("ticker")
    }
    
    if len(messages) < 2:
        return context  
    
    continuation_words = ["also", "and", "now", "then", "next", "additionally", "furthermore"]
    context["is_continuation"] = any(word in current_query for word in continuation_words)
    
    comparison_words = ["versus", "vs", "compared to", "compare", "difference between", "better than"]
    context["is_comparison"] = any(word in current_query for word in comparison_words)
    
    if context["is_continuation"] or context["is_comparison"]:
        context["is_new_topic"] = False
    
    return context


def handle_comparative_query(state: FinanceAgentState) -> Optional[Tuple[str, str]]:
    query = state["user_query"]
    
    comparison_pattern = r'(\w+)\s+(?:vs|versus|compared to|against)\s+(\w+)'
    match = re.search(comparison_pattern, query, re.IGNORECASE)
    
    if match:
        company1 = match.group(1)
        company2 = match.group(2)
        
        print(f"📊 Comparative query detected: {company1} vs {company2}")
        return (company1, company2)
    
    return None


async def store_file_dual(session_id: str, file_path: str, file_type: str, file_url: str):
    try:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            file_binary = f.read()
        
        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        binary_doc = await db.file_binaries.insert_one({
            "session_id": session_id,
            "type": file_type,
            "filename": filename,
            "binary_data": file_binary,
            "mime_type": mime_type,
            "size_bytes": len(file_binary),
            "created_at": datetime.now()
        })
        
        # Store file metadata
        await db.files.insert_one({
            "session_id": session_id,
            "type": file_type,
            "path": file_url,  
            "name": filename,
            "binary_id": binary_doc.inserted_id,  
            "size_bytes": len(file_binary),
            "mime_type": mime_type,
            "created_at": datetime.now()
        })
        
        print(f"✓ Stored {file_type} in both filesystem and MongoDB: {filename}")
        return binary_doc.inserted_id
        
    except Exception as e:
        print(f"✗ Error storing file: {e}")
        return None

def non_financial_query_node(state: FinanceAgentState) -> dict:
    response = "🤖 **Sorry!** My focus is financial analysis. Please ask me about a specific company's stock, news, or SEC filings (e.g., 'Show me Apple's stock chart' or 'What are the risks in Microsoft's 10-K?')."
    
    return {
        "final_answer": response,
        "user_friendly_message": response,
        "messages": [AIMessage(content=response)]
    }

# Helper functions
def check_for_chart_keywords(query: str) -> bool:
    query = query.lower()
    return any(keyword in query for keyword in ["chart", "plot", "graph", "visualize", "visualise"])

NON_FINANCIAL_KEYWORDS = [
    "date", "time", "weather", "capital of", "who is", 
    "meaning of", "how to cook", "recipe", "movie", "song", 
    "game", "joke", "poem", "story"
]

def is_non_financial_query(q):
    if not re.search(r"\b[A-Z]{2,6}\b", q):
        if any(word in q for word in NON_FINANCIAL_KEYWORDS):
            return True
    return False


def classify_intent(state: FinanceAgentState) -> dict:
   
    if llm is None:
        query = state["user_query"].lower()
        if any(word in query for word in ["chart", "plot", "graph", "price", "stock"]):
            return {"intent": "get_stock_data_and_chart"}
        elif any(word in query for word in ["news", "article", "headline"]):
            return {"intent": "get_financial_news"}
        elif any(word in query for word in ["filing", "10-k", "10-q", "sec"]):
            return {"intent": "get_sec_filing_section"}
        elif any(word in query for word in ["report", "analysis", "comprehensive"]):
            return {"intent": "get_report"}
        else:
            return {"intent": "greeting_help"}
    
    user_query = state["user_query"]
    query_lower = user_query.lower()
    messages = state.get("messages", [])

    if is_non_financial_query(query_lower):
        print("⚠ Non-financial query detected — routing to non_financial_query")
        return {"intent": "non_financial_query"}
    
    greeting_keywords = ["hello", "hi", "hey"]
    is_first_message = len(messages) <= 1  
    
    if is_first_message and any(keyword == query_lower.strip() for keyword in greeting_keywords):
        return {"intent": "greeting_help"}
    
    conversational_keywords = [
        "what can you do", "what can u do", "help", "help me", 
        "how do you work", "what are you", "who are you",
        "thanks", "thank you", "okay", "ok", "cool", "great",
        "tell me more", "explain", "how does this work"
    ]
    
    is_conversational = any(phrase in query_lower for phrase in conversational_keywords)
    has_no_ticker = not any(char.isupper() for char in user_query)  
    
    if is_conversational and has_no_ticker and len(query_lower.split()) < 15:
        return {"intent": "conversational_llm"}  
    
    chart_keywords = ["chart", "plot", "graph", "visualize", "visualise", "show me a chart"]
    if any(keyword in query_lower for keyword in chart_keywords):
        return {"intent": "get_stock_data_and_chart", "create_chart": True}
    
    news_keywords = ["news", "recent news", "latest news", "what's happening with", "headlines"]
    if any(keyword in query_lower for keyword in news_keywords):
        return {"intent": "get_financial_news"}
    
    metric_keywords = ["p/e ratio", "pe ratio", "market cap", "current price", "stock price"]
    if any(keyword in query_lower for keyword in metric_keywords):
        return {"intent": "get_stock_data_and_chart", "create_chart": False}
    
    report_keywords = ["generate report", "full report", "analyst report", "comprehensive analysis", "create a report"]
    if any(keyword in query_lower for keyword in report_keywords):
        return {"intent": "get_report"}
    
    
    section_keywords = {
        "risk": "Item 1A",
        "risks": "Item 1A", 
        "risk factors": "Item 1A",
        "business": "Item 1",
        "properties": "Item 2",
        "legal proceedings": "Item 3",
        "management discussion": "Item 7",
        "md&a": "Item 7",
        "financial statements": "Item 8",
        "controls and procedures": "Item 9A",
        "directors": "Item 10",
        "executive compensation": "Item 11",
        "security ownership": "Item 12",
        "exhibits": "Item 15"
    }
    
    
    section_patterns = [
        r"item\s+\d+[a-z]?",  # "Item 1A", "Item 7"
        r"section\s+\d+",
        r"part\s+[iv]+",
    ]
    
    
    has_section_keyword = any(keyword in query_lower for keyword in section_keywords.keys())
    has_section_pattern = any(re.search(pattern, query_lower) for pattern in section_patterns)
    

    section_request_phrases = [
        "extract",
        "show me",
        "get the",
        "pull",
        "retrieve",
        "find the",
        "display",
        "what are the risks",  
        "what risks",
        "list the risks",
        "show risks"
    ]
    
    has_section_request_phrase = any(phrase in query_lower for phrase in section_request_phrases)
    
    if (has_section_keyword or has_section_pattern) and has_section_request_phrase:
        return {"intent": "get_sec_filing_section"}
    
    history_text = ""
    for msg in state["messages"][-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"
    
    few_shot_prompt = f"""Conversation so far:
{history_text}

Stored context:
- last intent: {state.get("intent")}
- last company: {state.get("company_name")}
- last ticker: {state.get("ticker")}

User's latest message:
{user_query}

You are an expert at routing a user's query about financial analysis to the correct tool.

Your only job is to return a valid JSON object with a single key, "step", which indicates the correct tool to use.

The available tools are:
- greeting_help: ONLY for initial "Hi"/"Hello" (first message)
- conversational_llm: For follow-up questions, thanks, help requests, small talk
- get_sec_filing_section: Extract a COMPLETE SECTION by its official name (Item 1A, Item 7, etc.) OR when user asks "What are the risks" (they want the full risks section)
- get_financial_news: Get recent news articles about a company
- get_report: Generate a comprehensive analyst report with multiple data sources
- greeting_help: Respond to greetings or general help requests
- get_stock_data_and_chart: Get stock metrics, prices, and charts
- rag_filing_lookup: Answer SPECIFIC ANALYTICAL QUESTIONS by searching through filing content using semantic search

CRITICAL DISTINCTION:
- Use "get_sec_filing_section" when user wants:
  * An ENTIRE SECTION by name (Item 1A, Item 7, Section X)
  * "What are the risks" or "Show me the risks" (they want the full risks section)
  * "List the risks" or "Display risk factors"
  * ANY request that asks for a known section's content

- Use "rag_filing_lookup" when user asks:
  * Analytical questions requiring interpretation (How, Why, Compare, Analyze)
  * Questions that need synthesis across multiple sections
  * Questions that require semantic search (not just section extraction)

Examples:

Section Extraction (get_sec_filing_section):
Query: "Show me Item 1A from Apple's 10-K"
JSON: {{"step": "get_sec_filing_section"}}

Query: "What are the risks in Microsoft's latest 10-K filing?"
JSON: {{"step": "get_sec_filing_section"}}

Query: "Extract the risk factors section from Microsoft's filing"
JSON: {{"step": "get_sec_filing_section"}}

Query: "Get the business description from Tesla's 10-K"
JSON: {{"step": "get_sec_filing_section"}}

Query: "Show me the MD&A section"
JSON: {{"step": "get_sec_filing_section"}}

RAG Lookup (rag_filing_lookup):
Query: "How does Microsoft describe their cloud strategy compared to competitors?"
JSON: {{"step": "rag_filing_lookup"}}

Query: "Why is Tesla concerned about battery supply chains?"
JSON: {{"step": "rag_filing_lookup"}}

Query: "Explain Apple's AI strategy based on their filings"
JSON: {{"step": "rag_filing_lookup"}}

Query: "Compare Amazon's revenue growth to their risk disclosures"
JSON: {{"step": "rag_filing_lookup"}}

News:
Query: "Summarize recent news about Tesla"
JSON: {{"step": "get_financial_news"}}

Report:
Query: "Generate a full analyst report for Salesforce"
JSON: {{"step": "get_report"}}

Stock Data:
Query: "What is the P/E ratio for NVIDIA?"
JSON: {{"step": "get_stock_data_and_chart"}}

Query: "Show me a stock chart for Apple"
JSON: {{"step": "get_stock_data_and_chart"}}

Now classify this query. Respond with ONLY valid JSON:
Query: {user_query}
JSON:
"""
    
    try:
        if not llm:
            raise ValueError("LLM not initialized")
            
        response_str = llm.invoke(few_shot_prompt)
        
        if not response_str or not response_str.strip():
            raise ValueError("Empty response from LLM")
        
        response_str = response_str.strip()
        if response_str.startswith("```"):
            response_str = re.sub(r'```(?:json)?\s*|\s*```', '', response_str).strip()
        
        decision_json = json.loads(response_str)
        intent = decision_json.get("step", "")
        
        # Validate intent
        valid_intents = [
            "get_sec_filing_section",
            "get_financial_news", 
            "get_report",
            "greeting_help",
            "get_stock_data_and_chart",
            "rag_filing_lookup",
            "conversational_llm" 
        ]
        
        if intent not in valid_intents:
            print(f"⚠️ Invalid intent from LLM: {intent}. Using fallback.")
            if has_section_keyword:
                intent = "get_sec_filing_section"
            
            else:
                intent = "non_financial_query" 
        
        print(f"✓ Classified intent: {intent}")
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        print(f"Raw response: {response_str[:200] if response_str else 'None'}")
        if has_section_keyword:
            intent = "get_sec_filing_section"
        else:
           intent = "non_financial_query" 

    except Exception as e:
        print(f"❌ Generic error during classification: {e}")
        # Safe fallback
        if has_section_keyword:
            intent = "get_sec_filing_section"
        else:#
           intent = "non_financial_query" 
    updates = {"intent": intent}
    
    if intent == "get_stock_data_and_chart":
        is_chart_requested = check_for_chart_keywords(user_query)
        updates["create_chart"] = is_chart_requested
    
    return updates

def merge_entities(state, updates):
    if updates is None:
        return state

    if updates.get("company_name"):
        state["company_name"] = updates["company_name"]

    if updates.get("ticker"):
        state["ticker"] = updates["ticker"]

    if updates.get("intent"):
        state["intent"] = updates["intent"]
    
    # Handle chart flag
    if "create_chart" in updates:
        state["create_chart"] = updates["create_chart"]
    
    # Handle filing type and section
    if updates.get("filing_type"):
        state["filing_type"] = updates["filing_type"]
    
    if updates.get("section"):
        state["section"] = updates["section"]
    
    if updates.get("time_period"):
        state["time_period"] = updates["time_period"]

    return state

def greeting_help_node(state: FinanceAgentState) -> dict:
    
    instructions = """Hello! 😉

I am FinSight 💰📈 - your personal Financial Analyst ...

Here are some things you can ask me:
- Generate a stock performance chart for Amazon (AMZN)
- What are the risks in Microsoft's latest 10-K filing?
- Summarize recent news about Tesla (TSLA)
- What is the current P/E ratio for NVIDIA (NVDA)?
- Generate a full analyst report for Salesforce (CRM)
- How does Microsoft describe their cybersecurity risks?"""
    
    return {
        "final_answer": instructions,
        "user_friendly_message": instructions,
        "messages": [AIMessage(content=instructions)]
    }

def conversational_llm_node(state: FinanceAgentState) -> dict:
    """Fallback conversational responses using LLM for natural dialogue"""
    
    # Build conversation history
    history = ""
    for msg in state["messages"][-6:]:  # Last 6 messages for context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history += f"{role}: {msg.content}\n"
    
    system_prompt = f"""You are FinSight 💰📈, a friendly financial analysis AI assistant.

YOUR CAPABILITIES:
- Generate stock performance charts (need ticker symbol like AAPL, TSLA)
- Get current stock data: price, P/E ratio, market cap
- Fetch recent financial news about companies
- Extract SEC filing sections (10-K, 10-Q risk factors, business description, etc.)
- Answer analytical questions about SEC filings using semantic search
- Generate comprehensive analyst reports combining all data

CONVERSATION RULES:
- Be warm, conversational, and helpful
- Keep responses brief (2-4 sentences max)
- If user asks about a company, remind them to include ticker symbol
- Never make up stock prices, financial data, or news
- Guide users toward your capabilities naturally
- If they thank you or make small talk, respond naturally then suggest next steps
- For vague questions, ask for specific company/ticker

Previous conversation:
{history}

User's latest message: {state["user_query"]}

Respond naturally and guide them to use your features:"""
    
    try:
        if llm:
            llm_conversational = OllamaLLM(model="llama3.2")  
            response = llm_conversational.invoke(system_prompt)
            
            response = response.strip()
            
            print(f"💬 Conversational LLM response: {response[:100]}...")
            
            return {
                "final_answer": response,
                "user_friendly_message": response,
                "messages": [AIMessage(content=response)]
            }
        else:
            fallback = "I'm here to help with financial analysis! Ask me about any publicly traded company (include the ticker symbol like AAPL or TSLA)."
            return {
                "final_answer": fallback,
                "user_friendly_message": fallback,
                "messages": [AIMessage(content=fallback)]
            }
            
    except Exception as e:
        print(f"⚠️ Conversational LLM error: {e}")
        fallback = "I'm your financial analyst! Try asking me about a specific company's stock, news, or SEC filings."
        return {
            "final_answer": fallback,
            "user_friendly_message": fallback,
            "messages": [AIMessage(content=fallback)]
        }
    
def create_user_friendly_message(intent: str, state: dict) -> str:
    company_name = state.get("company_name", "the company")
    ticker = state.get("ticker", "")
    
    if intent == "get_stock_data_and_chart":
        metrics = state.get("structured_data") or {}
        current_price = metrics.get("current_price", "N/A")
        pe_ratio = metrics.get("pe_ratio", "N/A")
        market_cap = metrics.get("market_cap", "N/A")
        
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = str(market_cap)
        
        if state.get("create_chart"):
            message = f"📊 Here's the stock performance for **{company_name}** ({ticker}):\n\n"
        else:
            message = f"📈 Here are the key metrics for **{company_name}** ({ticker}):\n\n"
        
        message += f"💵 **Current Price:** ${current_price}\n"
        message += f"📊 **Market Cap:** {market_cap_str}\n"
        message += f"📉 **P/E Ratio:** {pe_ratio}\n"
        
        if state.get("data_range"):
            message += f"\n📅 Data Range: {state.get('data_range')}"
        
        return message
    
    elif intent == "get_financial_news":
        news_results = state.get("news_results", [])
        if news_results:
            message = f"📰 I found **{len(news_results)}** recent news articles about **{company_name}**. Here's what's happening:"
        else:
            message = f"📰 No recent news found for {company_name}."
        return message
    
    elif intent == "get_sec_filing_section":
        filing_type = state.get("filing_type", "10-K")
        section = state.get("section", "")
        tool_result = state.get("tool_result", "")
        
        if not tool_result:
            return f"📋 Sorry, I couldn't retrieve the {filing_type} filing for **{company_name}**. The filing might not be available or there was an error accessing it."
        preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
        
        message = f"📋 Here's what I found in **{company_name}'s** latest **{filing_type}** filing:\n\n{preview}"
        return message
    
    elif intent == "get_report":
        message = f"📊 I've generated a comprehensive financial analysis report for **{company_name}**.\n\n"
        message += "The report includes:\n"
        message += "• Key business risks from SEC filings\n"
        message += "• Recent market performance and metrics\n"
        message += "• Latest news and market catalysts\n\n"
        message += "You can download the full report below."
        return message
    
    elif intent == "rag_filing_lookup":
      return state.get("rag_answer", "I could not find any matching filing information.")

    
    return state.get("final_answer", "I've processed your request.")


async def rag_filing_lookup_node(state: FinanceAgentState) -> dict:
    try:
        ticker = state.get("ticker")
        query = state.get("user_query")
        
        
        if not ticker:
            error_message = "I need a company ticker symbol (e.g., AAPL, MSFT) to search their SEC filings. Please include a ticker in your question."
            print(f"✗ RAG lookup failed: No ticker provided")
            return {
                "rag_answer": None,
                "final_answer": error_message,
                "user_friendly_message": error_message,
                "messages": []
            }
        
        print(f"🔍 Running RAG query for {ticker}...")
        print(f"   Query: {query[:100]}...")
        
        print(f"🔍 Running RAG query for {ticker}...")
        print(f"   Query: {query[:100]}...")

        
        rag_answer = run_rag_query(query, ticker)
        
        
        if not rag_answer or rag_answer.strip() == "":
            error_message = f"I couldn't find relevant information in {ticker}'s SEC filings for your query."
            print(f"⚠️  RAG returned empty answer")
            return {
                "rag_answer": None,
                "final_answer": error_message,
                "user_friendly_message": error_message,
                "messages": []
            }
        
        
        if rag_answer.startswith("Error:") or "error" in rag_answer.lower()[:50]:
            print(f"⚠️  RAG returned error: {rag_answer[:100]}")
            return {
                "rag_answer": rag_answer,
                "final_answer": rag_answer,
                "user_friendly_message": rag_answer,
                "messages": []
            }
        
        print(f"✓ RAG returned {len(rag_answer)} characters")
        
        formatted_answer = f"📄 Based on {ticker}'s SEC filings:\n\n{rag_answer}"
        
        return {
            "rag_answer": rag_answer,
            "final_answer": formatted_answer,
            "user_friendly_message": formatted_answer,
            "messages": []
        }
        
    except FileNotFoundError as e:
        error_message = f"I couldn't find SEC filing documents for {state.get('ticker', 'this company')}. The filing may not be available in my database."
        print(f"✗ File not found: {e}")
        return {
            "rag_answer": None,
            "final_answer": error_message,
            "user_friendly_message": error_message,
            "messages": []
        }
        
    except Exception as e:
        error_message = f"I encountered an error while searching the SEC filings: {str(e)}"
        print(f"✗ RAG error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "rag_answer": None,
            "final_answer": error_message,
            "user_friendly_message": error_message,
            "messages": []
        }


async def process_query(state: FinanceAgentState) -> FinanceAgentState:
    print("\n" + "="*60)
    print("🗣️  CONVERSATIONAL PROCESSING")
    print("="*60)
    
    print("📊 Step 0: Analyzing conversation context...")
    context = analyze_conversation_context(state)
    print(f"   Is continuation: {context['is_continuation']}")
    print(f"   Is comparison: {context['is_comparison']}")
    print(f"   Previous: {context['previous_company']} ({context['previous_ticker']})")
    
    print("🔄 Step 0.5: Resolving coreferences...")
    original_query = state["user_query"]
    resolved_query = resolve_coreferences(original_query, state)
    if resolved_query != original_query:
        state["user_query"] = resolved_query
        state["original_query"] = original_query 
    
    print("🔍 Step 1: Extracting entities (context-aware)...")
    entities = extract_entities_with_context(state)
    state = merge_entities(state, entities)
    print(f"   Ticker: {state.get('ticker', 'N/A')}")
    print(f"   Company: {state.get('company_name', 'N/A')}")
    
    if context["is_comparison"]:
        print("⚖️  Step 2: Handling comparative query...")
        comparison = handle_comparative_query(state)
        if comparison:
            state["comparison_entities"] = comparison
            state["intent"] = "compare_entities"  
    
    print("🎯 Step 3: Classifying intent (context-aware)...")
    intent_update = await asyncio.to_thread(classify_intent, state)
    state.update(intent_update)
    
    intent = state.get("intent")
    print(f"   Intent: {intent}")
    
    print(f"🚀 Step 4: Executing {intent}...")
    
    if intent == "greeting_help":
        result = await asyncio.to_thread(greeting_help_node, state)
        state.update(result)
        state["user_friendly_message"] = state.get("final_answer")

    elif intent == "conversational_llm": 
        result = await asyncio.to_thread(conversational_llm_node, state)
        state.update(result)
        state["user_friendly_message"] = state.get("final_answer")

    elif intent == "non_financial_query": 
        result = await asyncio.to_thread(non_financial_query_node, state)
        state.update(result)
        state["user_friendly_message"] = state.get("final_answer")
        
    elif intent == "get_stock_data_and_chart":
        result = await asyncio.to_thread(get_stock_data_and_chart_node, state)
        state.update(result)
        friendly_message = create_user_friendly_message(intent, state)
        state["user_friendly_message"] = friendly_message
        
    elif intent == "get_financial_news":
        result = await asyncio.to_thread(get_financial_news_node, state)
        state.update(result)
        friendly_message = create_user_friendly_message(intent, state)
        state["user_friendly_message"] = friendly_message
        
    elif intent == "get_sec_filing_section":
        result = await asyncio.to_thread(get_sec_filing_section_node, state)
        state.update(result)
        friendly_message = create_user_friendly_message(intent, state)
        state["user_friendly_message"] = friendly_message
        
    elif intent == "get_report":
        stock_result = await asyncio.to_thread(get_stock_data_and_chart_node, state)
        state.update(stock_result)
        
        news_result = await asyncio.to_thread(get_financial_news_node, state)
        state.update(news_result)
        
        state["filing_type"] = "10-K"
        state["section"] = "1A"  
        print(f"📋 Report: Requesting filing_type={state['filing_type']}, section={state['section']}")
        
        sec_result = await asyncio.to_thread(get_sec_filing_section_node, state)
        state.update(sec_result)
        
        if sec_result.get("tool_result"):
            print(f"✅ Report: Risk data retrieved ({len(str(sec_result.get('tool_result')))} chars)")
        else:
            print(f"⚠️ Report: No risk data retrieved")
        
        report_result = await asyncio.to_thread(curate_report_node, state)
        state.update(report_result)
        
        friendly_message = create_user_friendly_message(intent, state)
        state["user_friendly_message"] = friendly_message
            
    elif intent == "rag_filing_lookup":
        state.update(await rag_filing_lookup_node(state))
    
    print(f"✓ Processing complete\n")
    return state

async def get_or_create_session(session_id: str):
    session = await db.sessions.find_one({"session_id": session_id})

    if session:
        return session_id, session

    initial_state = {
        "user_query": None,
        "messages": [], 
        "should_continue": True,
        "create_chart": False,
        "company_name": None,
        "ticker": None,
        "filing_type": None,
        "section": None,
        "tool_result": None,
        "structured_data": None,
        "final_answer": None,
        "report_data": None,
        "price_history_json": None,
        "news_results": None,
        "time_period": None,
        "intent": None,
        "user_friendly_message": None, 
        "chart_path": None,  
        "filing_info": None, 
        "data_range": None,  
         "rag_answer": None,
    }

    new_session = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "files": [],
        "state": initial_state
    }

    await db.sessions.insert_one(new_session)
    print(f"✓ Created new session: {session_id}")
    return session_id, new_session

@app.get("/api/root")
async def root():
    """API root endpoint"""
    return {
        "message": "FinSight API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/{session_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    try:
        count = await db.sessions.count_documents({})
        return {
            "status": "healthy",
            "ollama_status": "connected" if llm else "disconnected",
            "active_sessions": count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    try:
        cursor = db.files.find({"session_id": session_id})
        files = await cursor.to_list(length=5000)
        
        
        for file in files:
            file["_id"] = str(file["_id"])
            if "binary_id" in file:
                file["binary_id"] = str(file["binary_id"])

        return {
            "session_id": session_id,
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()

    try:  
        session_id, session_data = await get_or_create_session(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "content": f"Failed to create session: {str(e)}"
        })
        await websocket.close()
        return

    await websocket.send_json({
        "type": "session_created",
        "session_id": session_id
    })

    try:
        
        greeting_state = greeting_help_node(session_data["state"])
        await websocket.send_json({
            "type": "message",
            "content": greeting_state["final_answer"],
            "intent": "greeting"
        })

        
        while True:
            raw_message = await websocket.receive_text()

            try:
                message_data = json.loads(raw_message)
            except:
                message_data = {"message": raw_message}

            user_message = message_data.get("message", "")

            if user_message.lower() in ["exit", "quit"]:
                await websocket.send_json({
                    "type": "message",
                    "content": "Goodbye 👋"
                })
                break

            state_dict = session_data["state"].copy()
            
            state_dict["user_query"] = user_message
            state_dict["messages"].append(HumanMessage(content=user_message))
            
            try:
                state_obj = FinanceAgentState(state_dict)
            except Exception:
                state_obj = FinanceAgentState()
                state_obj.update(state_dict)

            await websocket.send_json({
                "type": "status",
                "content": "Processing your request..."
            })

            try:
                updated_state_obj = await process_query(state_obj)
                try:
                    updated_state = dict(updated_state_obj)
                except Exception:
                    updated_state = updated_state_obj

                db_state = updated_state.copy()
                
                if "messages" in db_state and db_state["messages"]:
                    serializable_messages = []
                    for msg in db_state["messages"]:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            serializable_messages.append({
                                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                                "content": msg.content
                            })
                        else:
                            serializable_messages.append(msg)
                    db_state["messages"] = serializable_messages

                # Update database with serialized state
                await db.sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {"state": db_state}}
                )

                # Update in-memory session data with original state
                session_data["state"] = updated_state
                
                # Get intent
                intent = updated_state.get("intent")
                
                # Use user-friendly message if available
                display_message = updated_state.get("user_friendly_message") or updated_state.get("final_answer", "I couldn't process that request.")

                # Ensure it's a string, not None
                if not isinstance(display_message, str) or not display_message:
                    display_message = "I couldn't process that request."

                # Store conversation messages
                await db.messages.insert_many([
                    {
                        "session_id": session_id,
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now()
                    },
                    {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": display_message,
                        "timestamp": datetime.now()
                    }
                ])

                response_packet = {
                    "type": "message",
                    "content": display_message,
                    "intent": intent,
                    "data": {}
                }

                # Attach extra data depending on intent
                if intent == "get_stock_data_and_chart":
                    response_packet["data"]["metrics"] = updated_state.get("structured_data")
                    chart_path = updated_state.get("chart_path")
                    
                    if chart_path and os.path.exists(chart_path):
                        # Convert to URL path (handle both Windows and Unix paths)
                        chart_url = "/" + str(chart_path).replace("\\", "/")
                        response_packet["data"]["chart_url"] = chart_url
                        
                        # Store file in database
                        await store_file_dual(
                            session_id=session_id,
                            file_path=chart_path,
                            file_type="chart",
                            file_url=chart_url
                        )

                elif intent == "get_financial_news":
                    news = updated_state.get("news_results", [])
                    response_packet["data"]["news"] = news

                elif intent == "get_sec_filing_section":
                    response_packet["data"]["filing_content"] = updated_state.get("tool_result", "")
                    response_packet["data"]["filing_info"] = updated_state.get("filing_info", {})

                elif intent == "get_report":
                    response_packet["data"]["report"] = updated_state.get("report_data")
                    
                    # Look for generated report files
                    reports_dir = Path("./reports")
                    if reports_dir.exists():
                        report_files = sorted(
                            reports_dir.glob("*.docx"),
                            key=lambda x: x.stat().st_mtime,
                            reverse=True
                        )
                        if report_files:
                            latest_report = report_files[0]
                            report_url = f"/reports/{latest_report.name}"
                            response_packet["data"]["report_url"] = report_url
                            
                            # Store file in database
                            await store_file_dual(
                                session_id=session_id,
                                file_path=str(latest_report),
                                file_type="report",
                                file_url=report_url
                            )
                elif intent == "rag_filing_lookup":
                    response_packet["data"]["rag_answer"] = updated_state.get("rag_answer", "")

                # Send final agent message to UI
                await websocket.send_json(response_packet)

                if display_message:
                    session_data["state"]["messages"].append(
                        AIMessage(content=display_message)
                    )

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"WebSocket error for session {session_id}:")
                print(error_details)
                
                # Send sanitized error to client
                error_message = str(e) if not isinstance(e, (IOError, OSError)) else "An error occurred processing your request"
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing request: {error_message}"
                })

    except WebSocketDisconnect:
        print(f"✓ WebSocket disconnected for session {session_id}")

    except Exception as e:
        print(f"✗ WebSocket error for session {session_id}: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        result = await db.sessions.delete_one({"session_id": session_id})
        await db.messages.delete_many({"session_id": session_id})
        await db.files.delete_many({"session_id": session_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    try:
        result = await db.sessions.find_one({"session_id": session_id})
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cursor = db.messages.find({"session_id": session_id}).sort("timestamp", 1)
        messages = await cursor.to_list(length=5000)
        
        return {
            "session_id": session_id,
            "created_at": result["created_at"],
            "message_count": len(messages),
            "messages": [
                {"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]} 
                for m in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/api/files/binary/{file_id}")
async def get_file_binary(file_id: str):
    try:
        file_doc = await db.file_binaries.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        return Response(
            content=file_doc["binary_data"],
            media_type=file_doc.get("mime_type", "application/octet-stream"),
            headers={
                "Content-Disposition": f'attachment; filename="{file_doc["filename"]}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@app.get("/api/sessions/{session_id}/files/detailed")
async def get_session_files_detailed(session_id: str):
    """Get detailed file information for a session"""
    try:
        cursor = db.files.find({"session_id": session_id})
        files = await cursor.to_list(length=5000)
        
        for file in files:
            file["_id"] = str(file["_id"])
            if "binary_id" in file:
                file["binary_id"] = str(file["binary_id"])
                file["download_url"] = f"/api/files/binary/{file['binary_id']}"
        
        return {
            "session_id": session_id,
            "file_count": len(files),
            "total_size_bytes": sum(f.get("size_bytes", 0) for f in files),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

@app.post("/api/files/restore/{session_id}")
async def restore_files_from_db(session_id: str):
    try:
        cursor = db.files.find({"session_id": session_id})
        files = await cursor.to_list(length=5000)
        
        restored = []
        errors = []
        
        for file_meta in files:
            try:
                # Get binary data
                binary_doc = await db.file_binaries.find_one({"_id": file_meta["binary_id"]})
                if not binary_doc:
                    errors.append(f"Binary not found for {file_meta['name']}")
                    continue
                
                file_path = file_meta["path"].lstrip("/")
        
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
                with open(file_path, 'wb') as f:
                    f.write(binary_doc["binary_data"])
                
                restored.append(file_meta["name"])
                
            except Exception as e:
                errors.append(f"Error restoring {file_meta['name']}: {str(e)}")
        
        return {
            "session_id": session_id,
            "restored_count": len(restored),
            "restored_files": restored,
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.delete("/api/sessions/{session_id}/complete")
async def delete_session_complete(session_id: str):
    try:
        session_result = await db.sessions.delete_one({"session_id": session_id})
        messages_result = await db.messages.delete_many({"session_id": session_id})
        
        files_cursor = db.files.find({"session_id": session_id})
        files = await files_cursor.to_list(length=5000)
        
        binary_ids = [f["binary_id"] for f in files if "binary_id" in f]
        binaries_result = await db.file_binaries.delete_many({"_id": {"$in": binary_ids}})
        
        files_result = await db.files.delete_many({"session_id": session_id})
        
        for file_meta in files:
            try:
                file_path = file_meta["path"].lstrip("/")
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠ Warning: Could not delete {file_path}: {e}")
        
        if session_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "message": f"Session {session_id} completely deleted",
            "deleted": {
                "sessions": session_result.deleted_count,
                "messages": messages_result.deleted_count,
                "files": files_result.deleted_count,
                "binaries": binaries_result.deleted_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
@app.post("/auth/register", response_model=dict)
async def register(user: UserRegister):
    try:
        user_id = create_user(user.username, user.email, user.password)
        return {
            "message": "User created successfully",
            "user_id": user_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """Login user and return JWT token"""
    authenticated_user = authenticate_user(user.username, user.password)
    
    if not authenticated_user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user["username"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": authenticated_user["id"],
            "username": authenticated_user["username"],
            "email": authenticated_user["email"]
        }
    }

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

@app.post("/auth/logout")
async def logout():
    return {"message": "Logged out successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)