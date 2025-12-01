#  main_backend.py 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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

# Import your existing modules
from state import FinanceAgentState
from workflows.extract_entities import extract_entities_node
from workflows.get_stock_data_and_chart import get_stock_data_and_chart_node
from workflows.get_financial_news import get_financial_news_node
from workflows.get_sec_filing_section import get_sec_filing_section_node
from workflows.curate_report import curate_report_node
from fastapi.responses import FileResponse
from langchain_ollama import OllamaLLM

app = FastAPI(title="FinSight API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Create directories if missing
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static directories for serving files
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve index.html at the root explicitly
@app.get("/", response_class=FileResponse)
async def serve_index():
    # return FileResponse("static/index.html")
    return FileResponse("static/frontend.html")

# Session storage - stores conversation history per session
sessions: Dict[str, Dict[str, Any]] = {}

# Initialize LLM
try:
    llm = OllamaLLM(model="llama3", format="json")
except Exception as e:
    print(f"Error initializing Ollama: {e}")
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
    query = query.lower()
    return any(keyword in query for keyword in ["chart", "plot", "graph", "visualize", "visualise"])


def classify_intent(state: FinanceAgentState) -> dict:
# Check if LLM is available
    if llm is None:
        # Fallback to keyword-based classification
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
    
    # Build conversation context
    history_text = ""
    for msg in state["messages"][-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    few_shot_prompt = f"""
Conversation so far:
{history_text}

Stored context:
- last intent: {state.get("intent")}
- last company: {state.get("company_name")}
- last ticker: {state.get("ticker")}

User's latest message:
{user_query}

Decide correct tool. Return JSON: {{"step":"..."}}
Possible steps: greeting_help, get_stock_data_and_chart, get_financial_news, get_sec_filing_section, get_report
"""

    try:
        response_str = llm.invoke(
            state["messages"] + [HumanMessage(content=few_shot_prompt)]
        )
        decision_json = json.loads(response_str)
        intent = decision_json.get("step", "greeting_help")
    except json.JSONDecodeError as e:
        print(f"JSON decode error in classify_intent: {e}")
        intent = state.get("intent") or "greeting_help"
    except Exception as e:
        print(f"Error in classify_intent: {e}")
        intent = state.get("intent") or "greeting_help"

    return {"intent": intent}



def merge_entities(state, updates):
    # Safely merge new entities without overwriting previous ones with None
    if updates is None:
        return state

    if updates.get("company_name"):
        state["company_name"] = updates["company_name"]

    if updates.get("ticker"):
        state["ticker"] = updates["ticker"]

    if updates.get("intent"):
        state["intent"] = updates["intent"]
    
    # ADD THIS - Handle chart flag
    if "create_chart" in updates:
        state["create_chart"] = updates["create_chart"]
    
    # ADD THIS - Handle filing type and section
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
"""
    return {
        "final_answer": instructions,
        "messages": [AIMessage(content=instructions)]
    }

def create_user_friendly_message(intent: str, state: dict) -> str:
    """Create user-friendly messages based on intent and data"""
    company_name = state.get("company_name", "the company")
    ticker = state.get("ticker", "")
    
    if intent == "get_stock_data_and_chart":
        metrics = state.get("structured_data", {})
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
        tool_result = state.get("tool_result")
        
        # Handle None or empty result
        if not tool_result:
            return f"📋 Sorry, I couldn't retrieve the {filing_type} filing for **{company_name}**. The filing might not be available or there was an error accessing it."
        
        # Create a preview of the content (first 500 chars)
        preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
        
        message = f"📋 Here's what I found in **{company_name}'s** latest **{filing_type}** filing"
        if section:
            message += f" ({section})"
        message += f":\n\n{preview}"
        
        return message
    
    elif intent == "get_report":
        message = f"📊 I've generated a comprehensive financial analysis report for **{company_name}**.\n\n"
        message += "The report includes:\n"
        message += "• Key business risks from SEC filings\n"
        message += "• Recent market performance and metrics\n"
        message += "• Latest news and market catalysts\n\n"
        message += "You can download the full report below."
        return message
    
    return state.get("final_answer", "I've processed your request.")

async def process_query(state: FinanceAgentState) -> FinanceAgentState:
    """Process a query through the agent workflow"""
    
    # Run synchronous node functions in thread pool
    entities = await asyncio.to_thread(extract_entities_node, state)
    state = merge_entities(state, entities)
    
    # Classify intent (also run in thread if synchronous)
    intent_update = await asyncio.to_thread(classify_intent, state)
    state.update(intent_update)
    
    intent = state.get("intent")
    
    # Route based on intent
    if intent == "greeting_help":
        result = await asyncio.to_thread(greeting_help_node, state)
        state.update(result)
        
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
        # Execute report workflow
        stock_result = await asyncio.to_thread(get_stock_data_and_chart_node, state)
        state.update(stock_result)
        
        news_result = await asyncio.to_thread(get_financial_news_node, state)
        state.update(news_result)
        
        sec_result = await asyncio.to_thread(get_sec_filing_section_node, state)
        state.update(sec_result)
        
        report_result = await asyncio.to_thread(curate_report_node, state)
        state.update(report_result)
        
        friendly_message = create_user_friendly_message(intent, state)
        state["user_friendly_message"] = friendly_message

    return state

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict]:
    """Get existing session or create new one."""
    if session_id:
        if session_id in sessions:
            return session_id, sessions[session_id]
        # create session with provided id
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "files": [],  # Track files generated in this session
            "state": {
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
            }
        }
        return session_id, sessions[session_id]

    # no session_id -> generate one
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "created_at": datetime.now().isoformat(),
        "files": [],
        "state": {
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
        }
    }
    return new_session_id, sessions[new_session_id]

@app.get("/api/root")
async def root():
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
    return {
        "status": "healthy",
        "ollama_status": "connected" if llm else "disconnected",
        "active_sessions": len(sessions)
    }

@app.get("/api/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    """Get all files generated in a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "files": sessions[session_id].get("files", [])
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Always use get_or_create_session
    session_id, session_data = get_or_create_session(session_id)

    # Inform client of session
    await websocket.send_json({
        "type": "session_created",
        "session_id": session_id
    })

    try:
        # Send initial greeting only if no messages yet
        if not session_data["state"]["messages"]:
            greeting_state = greeting_help_node(session_data["state"])
            greeting_message = greeting_state.get("final_answer", "")  # Add .get() with default
            
            if greeting_message:  # Only send if we have content
                await websocket.send_json({
                    "type": "message",
                    "content": greeting_message,
                    "intent": "greeting"
                })
                
                # Add greeting to conversation history
                session_data["state"]["messages"].append(AIMessage(content=greeting_message))

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
            state_dict = session_data["state"]

            # Update the state
            state_dict["user_query"] = user_message
            state_dict["messages"].append(HumanMessage(content=user_message))

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

                session_data["state"] = updated_state

                # Get intent
                intent = updated_state.get("intent")
                
                # Use user-friendly message if available
                display_message = updated_state.get("user_friendly_message") or updated_state.get("final_answer", "I couldn't process that request.")
                
                # Ensure it's a string, not None
                if not isinstance(display_message, str) or not display_message:
                    display_message = "I couldn't process that request."

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
                        # Convert to URL path
                        chart_url = "/" + chart_path.replace("\\", "/")
                        response_packet["data"]["chart_url"] = chart_url
                        
                        # Track file in session
                        session_data["files"].append({
                            "type": "chart",
                            "path": chart_url,
                            "name": os.path.basename(chart_path),
                            "created_at": datetime.now().isoformat()
                        })

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
                            
                            # Track file in session
                            session_data["files"].append({
                                "type": "report",
                                "path": report_url,
                                "name": latest_report.name,
                                "created_at": datetime.now().isoformat()
                            })

                # Send final agent message to UI
                await websocket.send_json(response_packet)
                
                # Add assistant reply to the UPDATED state (not old state_dict)
                if display_message:  # Only add if we have content
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
                    "content": error_message
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")

    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up associated files"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Clean up files (optional - you may want to keep them)
    for file_info in session.get("files", []):
        file_path = file_info["path"].lstrip("/")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Could not delete file {file_path}: {e}")
    
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted"}

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    messages = []
    for msg in session["state"]["messages"]:
        messages.append({
            "role": "human" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        })
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "message_count": len(messages),
        "messages": messages
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)