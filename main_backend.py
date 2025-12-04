#  main_backend.py 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import uuid
import os
import base64
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import asyncio
from pathlib import Path
from database import db, init_db, close_db
from contextlib import asynccontextmanager

from state import FinanceAgentState
from workflows.extract_entities import extract_entities_node
from workflows.get_stock_data_and_chart import get_stock_data_and_chart_node
from workflows.get_financial_news import get_financial_news_node
from workflows.get_sec_filing_section import get_sec_filing_section_node
from workflows.curate_report import curate_report_node
from fastapi.responses import FileResponse
from langchain_ollama import OllamaLLM
from fastapi.staticfiles import StaticFiles
from bson import ObjectId
from fastapi.responses import Response
import mimetypes
from authentication import (
    UserRegister, UserLogin, Token,
    create_user, authenticate_user, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from datetime import timedelta
from workflows.rag_filing_analysis import run_rag_query

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

async def store_file_dual(session_id: str, file_path: str, file_type: str, file_url: str):
    """Store file in both filesystem and MongoDB"""
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
        
        # Store binary data
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
    
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories for serving files
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve index.html at the root explicitly
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

# Helper functions
def check_for_chart_keywords(query: str) -> bool:
    """Check if query contains chart-related keywords"""
    query = query.lower()
    return any(keyword in query for keyword in ["chart", "plot", "graph", "visualize", "visualise"])

# Fixed classify_intent and process_query functions
# Replace these in your main_backend.py

import json
import re
from typing import Dict, Any

def classify_intent(state: FinanceAgentState) -> dict:
    """Classify user intent using LLM with improved logic"""
    user_query = state["user_query"]
    query_lower = user_query.lower()
    
    # ============================================================================
    # RULE-BASED PRE-CLASSIFICATION (Fast path for obvious cases)
    # ============================================================================
    
    # 1. Greetings - catch early
    greeting_keywords = ["hello", "hi", "hey", "help me", "what can you do", "how do you work"]
    if any(keyword in query_lower for keyword in greeting_keywords) and len(query_lower.split()) < 10:
        return {
            "intent": "greeting_help",
            "messages": []
        }
    
    # 2. Chart requests - very explicit
    chart_keywords = ["chart", "plot", "graph", "visualize", "visualise", "show me a chart"]
    if any(keyword in query_lower for keyword in chart_keywords):
        return {
            "intent": "get_stock_data_and_chart",
            "create_chart": True,
            "messages": []
        }
    
    # 3. News requests - explicit keywords
    news_keywords = ["news", "recent news", "latest news", "what's happening with", "headlines"]
    if any(keyword in query_lower for keyword in news_keywords):
        return {
            "intent": "get_financial_news",
            "messages": []
        }
    
    # 4. Specific metrics - P/E, market cap, etc.
    metric_keywords = ["p/e ratio", "pe ratio", "market cap", "current price", "stock price"]
    if any(keyword in query_lower for keyword in metric_keywords):
        return {
            "intent": "get_stock_data_and_chart",
            "create_chart": False,
            "messages": []
        }
    
    # 5. Report generation - explicit
    report_keywords = ["generate report", "full report", "analyst report", "comprehensive analysis", "create a report"]
    if any(keyword in query_lower for keyword in report_keywords):
        return {
            "intent": "get_report",
            "messages": []
        }
    
    # 6. Section extraction - very specific patterns
    section_patterns = [
        r"item\s+\d+[a-z]?",  # "Item 1A", "Item 7"
        r"section\s+\d+",
        r"part\s+[iv]+",
    ]
    section_keywords = ["extract section", "show me item", "get item"]
    
    has_section_pattern = any(re.search(pattern, query_lower) for pattern in section_patterns)
    has_section_keyword = any(keyword in query_lower for keyword in section_keywords)
    
    if has_section_pattern or has_section_keyword:
        return {
            "intent": "get_sec_filing_section",
            "messages": []
        }
    
    # ============================================================================
    # LLM-BASED CLASSIFICATION (For ambiguous cases)
    # ============================================================================
    
    few_shot_prompt = f"""You are an expert at routing a user's query about financial analysis to the correct tool.

Your only job is to return a JSON object with a single key, "step", which indicates the correct tool to use.

The available tools are:
- get_sec_filing_section: Extract a COMPLETE SECTION by its official name (Item 1A, Item 7, etc.)
- get_financial_news: Get recent news articles about a company
- get_report: Generate a comprehensive analyst report with multiple data sources
- greeting_help: Respond to greetings or general help requests
- get_stock_data_and_chart: Get stock metrics, prices, and charts
- rag_filing_lookup: Answer SPECIFIC QUESTIONS by searching through filing content using semantic search

CRITICAL DISTINCTION:
- Use "get_sec_filing_section" when user wants an ENTIRE SECTION by name (Item 1A, Item 7, Section X)
- Use "rag_filing_lookup" when user asks a QUESTION about filing content (What, How, Why, Explain, etc.)

Examples:

Section Extraction (get_sec_filing_section):
Query: "Show me Item 1A from Apple's 10-K"
JSON: {{"step": "get_sec_filing_section"}}

Query: "Extract the risk factors section from Microsoft's filing"
JSON: {{"step": "get_sec_filing_section"}}

RAG Lookup (rag_filing_lookup):
Query: "What risks did Apple report in their 10-K?"
JSON: {{"step": "rag_filing_lookup"}}

Query: "How does Microsoft describe their cloud business in their filings?"
JSON: {{"step": "rag_filing_lookup"}}

Query: "Explain Tesla's battery technology strategy from their SEC filings"
JSON: {{"step": "rag_filing_lookup"}}

Query: "What does Amazon say about AWS growth in their annual report?"
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
        
        # Clean response - remove markdown code blocks if present
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
            "rag_filing_lookup"
        ]
        
        if intent not in valid_intents:
            print(f"⚠️  Invalid intent from LLM: {intent}. Using fallback.")
            # Intelligent fallback
            if any(word in query_lower for word in ["what", "how", "why", "explain", "describe"]):
                intent = "rag_filing_lookup"
            else:
                intent = "greeting_help"
        
        print(f"✓ Classified intent: {intent}")
        
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error: {e}")
        print(f"Raw response: {response_str[:200] if response_str else 'None'}")
        # Fallback based on question words
        if any(word in query_lower for word in ["what", "how", "why", "explain", "describe", "tell me"]):
            intent = "rag_filing_lookup"
        else:
            intent = "greeting_help"
    except Exception as e:
        print(f"✗ Intent classification error: {e}")
        # Safe fallback
        if any(word in query_lower for word in ["what", "how", "why"]):
            intent = "rag_filing_lookup"
        else:
            intent = "greeting_help"
    
    updates = {
        "intent": intent,
        "messages": [],
    }
    
    if intent == "get_stock_data_and_chart":
        is_chart_requested = check_for_chart_keywords(user_query)
        updates["create_chart"] = is_chart_requested
    
    return updates


async def rag_filing_lookup_node(state: FinanceAgentState) -> dict:
    """
    Dedicated node to run RAG query against SEC filings and update state.
    """
    try:
        ticker = state.get("ticker")
        query = state.get("user_query")
        
        # Validate ticker exists
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
        
        # Call RAG function
        rag_answer = run_rag_query(query, ticker)
        
        # Check if answer is valid
        if not rag_answer or rag_answer.strip() == "":
            error_message = f"I couldn't find relevant information in {ticker}'s SEC filings for your query."
            print(f"⚠️  RAG returned empty answer")
            return {
                "rag_answer": None,
                "final_answer": error_message,
                "user_friendly_message": error_message,
                "messages": []
            }
        
        # Check for error messages from RAG
        if rag_answer.startswith("Error:") or "error" in rag_answer.lower()[:50]:
            print(f"⚠️  RAG returned error: {rag_answer[:100]}")
            return {
                "rag_answer": rag_answer,
                "final_answer": rag_answer,
                "user_friendly_message": rag_answer,
                "messages": []
            }
        
        print(f"✓ RAG returned {len(rag_answer)} characters")
        
        # Format answer nicely
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
    """Process a query through the agent workflow"""
    
    # STEP 1: Extract entities FIRST (critical - must happen before classification)
    print("🔍 Step 1: Extracting entities...")
    state.update(extract_entities_node(state))
    print(f"   Ticker: {state.get('ticker', 'N/A')}")
    print(f"   Company: {state.get('company_name', 'N/A')}")
    
    # STEP 2: Classify intent
    print("🎯 Step 2: Classifying intent...")
    state.update(classify_intent(state))
    
    intent = state.get("intent")
    print(f"   Intent: {intent}")
    
    # STEP 3: Route based on intent
    print(f"🚀 Step 3: Executing {intent}...")
    
    if intent == "greeting_help":
        state.update(greeting_help_node(state))
        
    elif intent == "get_stock_data_and_chart":
        state.update(get_stock_data_and_chart_node(state))
        try:
            friendly_message = create_user_friendly_message(intent, state)
            state["user_friendly_message"] = friendly_message
        except Exception as e:
            print(f"✗ Error creating friendly message: {e}")
            state["user_friendly_message"] = state.get("final_answer", "I encountered an error processing your request.")
        
    elif intent == "get_financial_news":
        state.update(get_financial_news_node(state))
        try:
            friendly_message = create_user_friendly_message(intent, state)
            state["user_friendly_message"] = friendly_message
        except Exception as e:
            print(f"✗ Error creating friendly message: {e}")
            state["user_friendly_message"] = state.get("final_answer", "I encountered an error processing your request.")
        
    elif intent == "get_sec_filing_section":
        state.update(get_sec_filing_section_node(state))
        try:
            friendly_message = create_user_friendly_message(intent, state)
            state["user_friendly_message"] = friendly_message
        except Exception as e:
            print(f"✗ Error creating friendly message: {e}")
            state["user_friendly_message"] = state.get("final_answer", "I encountered an error processing your request.")
        
    elif intent == "get_report":
        # Execute report workflow
        state.update(get_stock_data_and_chart_node(state))
        state.update(get_financial_news_node(state))
        state.update(get_sec_filing_section_node(state))
        state.update(curate_report_node(state))
        try:
            friendly_message = create_user_friendly_message(intent, state)
            state["user_friendly_message"] = friendly_message
        except Exception as e:
            print(f"✗ Error creating friendly message: {e}")
            state["user_friendly_message"] = state.get("final_answer", "I encountered an error processing your request.")

    elif intent == "rag_filing_lookup":
        # RAG-based filing analysis
        state.update(await rag_filing_lookup_node(state))
        # user_friendly_message is already set by rag_filing_lookup_node
    
    print(f"✓ Step 3 complete")
    return state

def greeting_help_node(state: FinanceAgentState) -> dict:
    """Return greeting and help instructions"""
    instructions = """Hello! 😉

I am FinSight 💰📈 - your personal Financial Analyst ...

Here are some things you can ask me:
- Generate a stock performance chart for Amazon (AMZN)
- What are the risks in Microsoft's latest 10-K filing?
- Summarize recent news about Tesla (TSLA)
- What is the current P/E ratio for NVIDIA (NVDA)?
- Generate a full analyst report for Salesforce (CRM)
"""
    return {
        "final_answer": instructions,
        "messages": []
    }

def create_user_friendly_message(intent: str, state: dict) -> str:
    """Create user-friendly messages based on intent and data"""
    company_name = state.get("company_name", "the company")
    ticker = state.get("ticker", "")
    
    if intent == "get_stock_data_and_chart":
        metrics = state.get("structured_data") or {}
        current_price = metrics.get("current_price", "N/A")
        pe_ratio = metrics.get("pe_ratio", "N/A")
        market_cap = metrics.get("market_cap", "N/A")
        
        # Format market cap
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
        
        # Create a preview of the content (first 500 chars)
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


    """
    Dedicated node to run RAG query against SEC filings and update state.
    """
    try:
        # Check if ticker is available, essential for RAG
        if not state.get("ticker"):
            # Provide a fallback message if entity extraction failed
            error_message = "I need a company ticker (e.g., AAPL) to perform RAG analysis on a filing. Please try again."
            return {
                "rag_answer": None,
                "final_answer": error_message,
                "user_friendly_message": error_message,
                "messages": []
            }
            
        rag_answer = run_rag_query(state["user_query"], state["ticker"])
        return {
            "rag_answer": rag_answer,
            "final_answer": rag_answer,
            "user_friendly_message": rag_answer,
            "messages": []
        }
    except Exception as e:
        error_message = f"RAG lookup failed for {state.get('ticker', 'the company')}: {str(e)}"
        print(f"✗ {error_message}")
        return {
            "rag_answer": None,
            "final_answer": error_message,
            "user_friendly_message": error_message,
            "messages": []
        }

async def get_or_create_session(session_id: str):
    """Get existing session or create new one"""
    session = await db.sessions.find_one({"session_id": session_id})

    # If found → return it
    if session:
        return session_id, session

    # Otherwise → create new session
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
    """Health check endpoint"""
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
    """Get all files generated in a session"""
    try:
        cursor = db.files.find({"session_id": session_id})
        files = await cursor.to_list(length=5000)
        
        # Convert ObjectId to string
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
        # Send initial greeting
        greeting_state = greeting_help_node(session_data["state"])
        await websocket.send_json({
            "type": "message",
            "content": greeting_state["final_answer"],
            "intent": "greeting"
        })

        # Main loop
        while True:
            raw_message = await websocket.receive_text()

            try:
                message_data = json.loads(raw_message)
            except:
                message_data = {"message": raw_message}

            user_message = message_data.get("message", "")

            # Exit condition
            if user_message.lower() in ["exit", "quit"]:
                await websocket.send_json({
                    "type": "message",
                    "content": "Goodbye 👋"
                })
                break

            # Retrieve existing state dictionary
            state_dict = session_data["state"].copy()

            # Update the state
            state_dict["user_query"] = user_message

            # Convert to FinanceAgentState
            try:
                state_obj = FinanceAgentState(state_dict)
            except Exception:
                state_obj = FinanceAgentState()
                state_obj.update(state_dict)

            # Inform client of status
            await websocket.send_json({
                "type": "status",
                "content": "Processing your request..."
            })

            # Run through agent workflow
            try:
                updated_state_obj = await process_query(state_obj)

                # Convert back to plain dict for saving
                try:
                    updated_state = dict(updated_state_obj)
                except Exception:
                    updated_state = updated_state_obj

                # Create a deep copy for database storage
                db_state = updated_state.copy()
                
                # Convert LangChain message objects to serializable dicts for database only
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

            except Exception as e:
                import traceback
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing request: {str(e)}"
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
    """Delete a session and its messages/files metadata"""
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
    """Get conversation history for a session"""
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
    """Download file binary from MongoDB"""
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
        
        # Convert ObjectId to string for JSON serialization
        for file in files:
            file["_id"] = str(file["_id"])
            if "binary_id" in file:
                file["binary_id"] = str(file["binary_id"])
                # Add download URL for MongoDB binary
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
    """Restore files from MongoDB to filesystem"""
    try:
        # Get all files for this session
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
                
                # Reconstruct filesystem path from URL
                file_path = file_meta["path"].lstrip("/")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write file to filesystem
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
    """Completely delete a session including all files and binaries"""
    try:
        # Delete from all collections
        session_result = await db.sessions.delete_one({"session_id": session_id})
        messages_result = await db.messages.delete_many({"session_id": session_id})
        
        # Get file metadata to delete binaries
        files_cursor = db.files.find({"session_id": session_id})
        files = await files_cursor.to_list(length=5000)
        
        # Delete binary data
        binary_ids = [f["binary_id"] for f in files if "binary_id" in f]
        binaries_result = await db.file_binaries.delete_many({"_id": {"$in": binary_ids}})
        
        # Delete file metadata
        files_result = await db.files.delete_many({"session_id": session_id})
        
        # Optionally delete filesystem files
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
    """Register a new user"""
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
    """Get current user information"""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

@app.post("/auth/logout")
async def logout():
    """Logout endpoint (token invalidation handled on client side)"""
    return {"message": "Logged out successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)