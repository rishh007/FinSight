# workflows/extract_entities.py

import json
from langchain_ollama import OllamaLLM
from state import FinanceAgentState
from rich.markdown import Markdown as RichMarkdown

llm = OllamaLLM(model="llama3", format="json")

def extract_entities_node(state: FinanceAgentState) -> dict:
    """Extracts all necessary parameters (ticker, company name, filing type, and section) 
    from the user's query and updates the state."""
    

    print("---NODE: Extracting Entities---")
    user_query = state.get("user_query", "")
    
    prompt = f"""
    You are an expert at extracting financial entities from user queries.
    Your task is to extract four pieces of information from the query: ticker, company name, filing type, section code, and time period.
    
    Default value for time period is "1"
    "
    Map common terms to the following technical codes:
    - 'Annual report' or '10-K' should map to filing_type: '10-K'.
    - 'Quarterly report' or '10-Q' should map to filing_type: '10-Q'.
    - 'Risks' or 'risk factors' should map to section: '1A'.
    - 'Management discussion' or 'MD&A' should map to section: '7'.
    
    Return the result as a JSON object with keys "ticker", "company_name", "filing_type", "section" and "time_period".
    If a value is not found, return null for that key.

    Example 1 (Full Extraction):
    Query: "Pull the risk factors from Apple's latest 10-K."
    JSON: {{"ticker": "AAPL", "company_name": "Apple", "filing_type": "10-K", "section": "1A", "time_period": "1"}}

    Example 2 (Partial Extraction):
    Query: "Get me the latest news for Microsoft for the last 5 years."
    JSON: {{"ticker": "MSFT", "company_name": "Microsoft", "filing_type": null, "section": null, "time_period" : "5" "}}

    Query: "{user_query}"
    JSON:
    """
    
    try:
        response_str = llm.invoke(prompt)
        entities = json.loads(response_str)
        
        ticker = entities.get("ticker")
        company_name = entities.get("company_name")
        filing_type = entities.get("filing_type")
        section = entities.get("section")
        time_period = entities.get("time_period")

        if not ticker and not company_name:
            print("No ticker or company name found.")
            print(RichMarkdown("I couldn't identify the company in your query. Please be more specific."))
            return {"final_answer": "I couldn't identify the company in your query. Please be more specific."}

        print(f"Entities extracted: Ticker={ticker}, Filing={filing_type}, Section={section} and Time Period (default 1y) = {time_period}")
        
        return {
            "ticker": ticker, 
            "company_name": company_name,
            "filing_type": filing_type,
            "section": section,
            "time_period": time_period
        }
        
    except Exception as e:
        error_message = f"Failed to extract entities. Error: {e}"
        print(error_message)
        return {"final_answer": error_message}