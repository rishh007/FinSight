import os
from typing import TypedDict, Optional, List
from sec_api import ExtractorApi
from dotenv import load_dotenv
from state import FinanceAgentState

# Load environment variables from .env file
load_dotenv()

def get_sec_filing_section_node(state: FinanceAgentState) -> dict:
    """
    Extracts a specific section from the most recent SEC filing of a given type 
    for a company using data from the agent's state.
    """
    print("---NODE: Getting SEC Filing Section---")
# --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
    if state.get("final_answer"):
        print("\n" + state["final_answer"])
        print("-" * 30)
    # --------------------------------------------------
    # Retrieve necessary parameters from the state
    ticker = state.get("ticker")
    filing_type = state.get("filing_type")
    section = state.get("section")
    
    # Check for required parameters
    if not all([ticker, filing_type, section]):
        error_message = "Missing required parameters (ticker, filing_type, or section) in the state."
        print(f"Error: {error_message}")
        return {"final_answer": error_message}
    
    # Initialize the API client
    sec_api_key = os.getenv("SEC_API_KEY")
    if not sec_api_key:
        error_message = "SEC_API_KEY environment variable is not set."
        print(f"Error: {error_message}")
        return {"final_answer": error_message}
    
    extractorApi = ExtractorApi(sec_api_key)
    
    try:
        # URL of the latest filing of the specified type
        filing_url = extractorApi.get_filing_url(ticker=ticker, form_type=filing_type)
        

        
        if not filing_url:
            message = f"No recent {filing_type} found for {ticker}."
            print(message)
            return {"final_answer": message}
        
        # Extract the text of the specified section
        section_text = extractorApi.get_section(filing_url=filing_url, section=section, return_type="text")
        
        # Truncate for brevity and context limits
        truncated_text = section_text[:4000] if len(section_text) > 4000 else section_text
        
        # Update the state with the result
        success_message = f"Successfully extracted section {section} from the latest {filing_type} for {ticker}."
        print(success_message)
        
        return {
            "tool_result": truncated_text,
            "final_answer": success_message
        }
        
    except Exception as e:
        error_message = f"Could not extract section {section} from {filing_type} for {ticker}. Error: {e}"
        print(f"Error: {error_message}")
        return {"final_answer": error_message}