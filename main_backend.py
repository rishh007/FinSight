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

def classify_intent(state: FinanceAgentState) -> dict:
    """Classify user intent using LLM"""
    user_query = state["user_query"]
    
    few_shot_prompt = f"""You are an expert at routing a user's query about financial analysis to the correct tool.

Your only job is to return a JSON object with a single key, "step", which indicates the correct tool to use.

The available tools are:
- get_sec_filing_section
- get_financial_news
- get_report
- greeting_help
- get_stock_data_and_chart

Examples:

Queries for "get_sec_filing_section":
- Query: What were the business risks listed in Apple's (AAPL) most recent 10-K?
- JSON: {{"step": "get_sec_filing_section"}}

Queries for "get_financial_news":
- Query: Summarize recent news about Tesla (TSLA).
- JSON: {{"step": "get_financial_news"}}

Queries for "get_report":
- Query: Generate a full analyst report for Salesforce (CRM).
- JSON: {{"step": "get_report"}}

Queries for "greeting_help":
- Query: Hello, can you help me?
- JSON: {{"step": "greeting_help"}}

Queries for "get_stock_data_and_chart":
- Query: What is the current P/E ratio for NVIDIA (NVDA)?
- JSON: {{"step": "get_stock_data_and_chart"}}

Now, based on the user's query below, provide the JSON object.
Query: {user_query}
JSON:
"""
    
    try:
        if not llm:
            raise ValueError("LLM not initialized")
            
        response_str = llm.invoke(few_shot_prompt)
        
        if not response_str:
            raise ValueError("Empty response from LLM")
            
        decision_json = json.loads(response_str)
        intent = decision_json.get("step", "greeting_help") 
    except Exception as e:
        print(f"✗ Intent classification error: {e}")
        intent = "greeting_help"
    
    updates = {
        "intent": intent,
        "messages": [],
    }
    
    if intent == "get_stock_data_and_chart":
        is_chart_requested = check_for_chart_keywords(user_query)
        updates["create_chart"] = is_chart_requested
    
    return updates

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
    
    return state.get("final_answer", "I've processed your request.")

async def process_query(state: FinanceAgentState) -> FinanceAgentState:
    """Process a query through the agent workflow"""
    
    # Extract entities
    state.update(extract_entities_node(state))
    
    # Classify intent
    state.update(classify_intent(state))
    
    intent = state.get("intent")
    
    # Route based on intent
    if intent == "greeting_help":
        state.update(greeting_help_node(state))
        
    elif intent == "get_stock_data_and_chart":
        state.update(get_stock_data_and_chart_node(state))
        # Create user-friendly message
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
    
    return state

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)