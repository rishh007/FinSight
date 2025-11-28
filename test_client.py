# websocket client - to be paired with , main_backend.py

import asyncio
import websockets
import json
import sys

async def test_websocket():
    """Test the WebSocket connection to FinSight API"""
    
    session_id = "test-session-123"
    uri = f"ws://localhost:5500/ws/{session_id}"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!\n")
            
            # Receive initial greeting
            greeting_response = await websocket.recv()
            greeting_data = json.loads(greeting_response)
            print("📩 Received greeting:")
            print(f"Type: {greeting_data.get('type')}")
            print(f"Content:\n{greeting_data.get('content')}\n")
            print("-" * 60)
            
            # Test queries
            test_queries = [
                "What is the current stock price of Apple (AAPL)?",
                "Show me recent news about Tesla",
                "exit"
            ]
            
            for query in test_queries:
                print(f"\n📤 Sending: {query}")
                
                # Send query
                await websocket.send(json.dumps({"message": query}))
                
                if query.lower() == "exit":
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    print(f"📥 Response: {response_data.get('content')}")
                    break
                
                # Receive status update
                try:
                    status_response = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=2.0
                    )
                    status_data = json.loads(status_response)
                    if status_data.get('type') == 'status':
                        print(f"⏳ Status: {status_data.get('content')}")
                except asyncio.TimeoutError:
                    pass
                
                # Receive actual response
                response = await asyncio.wait_for(
                    websocket.recv(), 
                    timeout=60.0  # Longer timeout for processing
                )
                response_data = json.loads(response)
                
                print(f"\n📥 Response:")
                print(f"Type: {response_data.get('type')}")
                print(f"Intent: {response_data.get('intent')}")
                print(f"Content:\n{response_data.get('content')[:500]}...")
                
                # Show additional data if available
                if response_data.get('data'):
                    print(f"\n📊 Additional Data:")
                    data = response_data['data']
                    if 'metrics' in data:
                        print(f"  Metrics: {list(data['metrics'].keys())}")
                    if 'news' in data:
                        print(f"  News articles: {len(data['news'])} found")
                    if 'chart_path' in data:
                        print(f"  Chart saved: {data['chart_path']}")
                
                print("-" * 60)
                
                # Wait a bit between queries
                await asyncio.sleep(2)
            
            print("\n✅ Test completed successfully!")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the FastAPI server is running on port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def interactive_mode():
    """Interactive chat mode"""
    session_id = "interactive-session"
    uri = f"ws://localhost:5500/ws/{session_id}"
    
    print("🚀 Starting interactive mode...")
    print(f"Connecting to {uri}...\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Type 'exit' to quit.\n")
            
            # Receive greeting
            greeting_response = await websocket.recv()
            greeting_data = json.loads(greeting_response)
            print(f"Agent: {greeting_data.get('content')}\n")
            print("-" * 60)
            
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Send query
                await websocket.send(json.dumps({"message": user_input}))
                
                if user_input.lower() in ["exit", "quit"]:
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    print(f"\nAgent: {response_data.get('content')}")
                    break
                
                # Receive response
                while True:
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'status':
                        print(f"⏳ {response_data.get('content')}")
                    elif response_data.get('type') == 'message':
                        print(f"\nAgent: {response_data.get('content')}")
                        
                        # Show additional info
                        if response_data.get('data'):
                            data = response_data['data']
                            if 'chart_path' in data and data['chart_path']:
                                print(f"📊 Chart saved to: {data['chart_path']}")
                        
                        break
                    elif response_data.get('type') == 'error':
                        print(f"❌ Error: {response_data.get('content')}")
                        break
                
                print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        print("Running test mode. Use 'python test_client.py interactive' for chat mode.\n")
        asyncio.run(test_websocket())