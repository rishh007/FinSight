from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import uuid
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import asyncio

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

# Serve index.html at the root explicitly
@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("static/index.html")

from fastapi.staticfiles import StaticFiles

# Serve static files (your frontend)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


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

# Helper functions from original code
def check_for_chart_keywords(query: str) -> bool:
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
        response_str = llm.invoke(few_shot_prompt)
        decision_json = json.loads(response_str)
        intent = decision_json["step"]
    except Exception as e:
        print(f"Intent classification error: {e}")
        intent = "greeting_help"
    
    updates = {
        "intent": intent,
        "messages": [HumanMessage(content=user_query)],
    }
    
    if intent == "get_stock_data_and_chart":
        is_chart_requested = check_for_chart_keywords(user_query)
        updates["create_chart"] = is_chart_requested
    
    return updates

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
        
    elif intent == "get_financial_news":
        state.update(get_financial_news_node(state))
        
    elif intent == "get_sec_filing_section":
        state.update(get_sec_filing_section_node(state))
        
    elif intent == "get_report":
        # Execute report workflow
        state.update(get_stock_data_and_chart_node(state))
        state.update(get_financial_news_node(state))
        state.update(get_sec_filing_section_node(state))
        state.update(curate_report_node(state))
    
    return state

# def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict]:
#     """Get existing session or create new one"""
#     if session_id and session_id in sessions:
#         return session_id, sessions[session_id]
    
#     new_session_id = str(uuid.uuid4())
#     sessions[new_session_id] = {
#         "created_at": datetime.now().isoformat(),
#         "state": {
#             "user_query": None,
#             "messages": [],
#             "should_continue": True,
#             "create_chart": False,
#             "company_name": None,
#             "ticker": None,
#             "filing_type": None,
#             "section": None,
#             "tool_result": None,
#             "structured_data": None,
#             "final_answer": None,
#             "report_data": None,
#             "price_history_json": None,
#             "news_results": None,
#             "time_period": None,
#             "intent": None,
#         }
#     }
#     return new_session_id, sessions[new_session_id]
def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict]:
    """
    Get existing session or create new one.
    If client provided a session_id, create a session using that id if it doesn't exist.
    """
    if session_id:
        if session_id in sessions:
            return session_id, sessions[session_id]
        # create session with provided id
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
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


@app.get("/")
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


# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await websocket.accept()
    
#     # Get or create session
#     if session_id not in sessions:
#         session_id, session_data = get_or_create_session(session_id)
#         await websocket.send_json({
#             "type": "session_created",
#             "session_id": session_id
#         })
#     else:
#         session_data = sessions[session_id]
    
#     try:
#         # Send greeting
#         greeting_state = greeting_help_node(session_data["state"])
#         await websocket.send_json({
#             "type": "message",
#             "content": greeting_state["final_answer"],
#             "intent": "greeting"
#         })
        
#         while True:
#             # Receive message
#             data = await websocket.receive_text()
#             message_data = json.loads(data)
#             user_message = message_data.get("message", "")
            
#             if user_message.lower() in ["exit", "quit"]:
#                 await websocket.send_json({
#                     "type": "message",
#                     "content": "Goodbye 👋"
#                 })
#                 break
            
#             # Update state
#             state = session_data["state"]
#             state["user_query"] = user_message
#             state["messages"].append(HumanMessage(content=user_message))
            
#             # Send processing status
#             await websocket.send_json({
#                 "type": "status",
#                 "content": "Processing your request..."
#             })
            
#             # Process query
#             try:
#                 updated_state = await process_query(state)
#                 session_data["state"] = updated_state
                
#                 # Extract response
#                 final_answer = updated_state.get("final_answer", "I couldn't process that request.")
#                 intent = updated_state.get("intent")
                
#                 # Send response
#                 response_data = {
#                     "type": "message",
#                     "content": final_answer,
#                     "intent": intent,
#                     "data": {}
#                 }
                
#                 # Include relevant data based on intent
#                 if intent == "get_stock_data_and_chart":
#                     response_data["data"]["metrics"] = updated_state.get("structured_data")
#                     response_data["data"]["chart_path"] = updated_state.get("chart_path")
#                 elif intent == "get_financial_news":
#                     response_data["data"]["news"] = updated_state.get("news_results")
#                 elif intent == "get_report":
#                     response_data["data"]["report"] = updated_state.get("report_data")
                
#                 await websocket.send_json(response_data)
                
#             except Exception as e:
#                 await websocket.send_json({
#                     "type": "error",
#                     "content": f"Error processing request: {str(e)}"
#                 })
    
#     except WebSocketDisconnect:
#         print(f"WebSocket disconnected for session {session_id}")
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#         await websocket.close()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Always use get_or_create_session (fixes client/server mismatch)
    session_id, session_data = get_or_create_session(session_id)

    # Inform client of session (even if it already existed)
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
            # Wait for incoming WS message
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

            # Convert to FinanceAgentState (important for your workflow nodes)
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
                    updated_state = updated_state_obj.to_dict()

                session_data["state"] = updated_state

                # Prepare final answer
                final_answer = updated_state.get("final_answer", "I couldn't process that request.")
                intent = updated_state.get("intent")

                response_packet = {
                    "type": "message",
                    "content": final_answer,
                    "intent": intent,
                    "data": {}
                }

                # Attach extra data depending on intent
                if intent == "get_stock_data_and_chart":
                    response_packet["data"]["metrics"] = updated_state.get("structured_data")
                    response_packet["data"]["chart_path"] = updated_state.get("chart_path")

                elif intent == "get_financial_news":
                    response_packet["data"]["news"] = updated_state.get("news_results")

                elif intent == "get_report":
                    response_packet["data"]["report"] = updated_state.get("report_data")

                # Send final agent message to UI
                await websocket.send_json(response_packet)

            except Exception as e:
                # Workflow crashed
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing request: {str(e)}"
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
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

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

# from fastapi import FastAPI, WebSocket
# from fastapi.staticfiles import StaticFiles
# # Serve static files (HTML, JS, CSS)
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     await websocket.send_text("WebSocket connected!")
#     while True:
#         try:
#             msg = await websocket.receive_text()
#             await websocket.send_text(f"Echo: {msg}")
#         except:
#             break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)