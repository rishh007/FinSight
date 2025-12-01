# workflows/extract_entities.py

import json
from langchain_ollama import OllamaLLM
from state import FinanceAgentState
from rich.markdown import Markdown as RichMarkdown
from langchain_core.messages import AIMessage, HumanMessage 

llm = OllamaLLM(model="llama3", format="json")

def map_section_code(section):
    """Maps shorthand section codes to their full SEC API identifiers"""
    section_mapping = {
        "1A": ["part2item1a", "part1item1"],
        "7": "part2item7",
        "1": "part1item1",
        "2": "part2item2",
    }
    result = section_mapping.get(section, section)
    return result[0] if isinstance(result, list) else result

def extract_entities_node(state: FinanceAgentState) -> dict:
    """Extracts all necessary parameters (ticker, company name, filing type, section, create_chart) 
    from the user's query and updates the state."""
    
    print("---NODE: Extracting Entities---")
    user_query = state.get("user_query", "")
    
    # Check for chart keywords FIRST (before LLM call)
    chart_keywords = ["chart", "plot", "graph", "visualize", "visualise", "show me", "display"]
    create_chart = any(keyword in user_query.lower() for keyword in chart_keywords)
    
    prompt = f"""
    You are an expert at extracting financial entities from user queries.
    Your task is to extract these pieces of information: ticker, company name, filing type, section code, and time period.
    
    Default value for time period is "1"
    
    Map common terms to technical codes:
    - 'Annual report' or '10-K' → filing_type: '10-K'
    - 'Quarterly report' or '10-Q' → filing_type: '10-Q'
    - 'Risks' or 'risk factors' → section: 'part2item1a'
    - 'Management discussion' or 'MD&A' → section: 'part2item7'
    - 'Business description' → section: 'part1item1'
    - 'Financial information' → section: 'part2item2'
    
    Return JSON with keys: "ticker", "company_name", "filing_type", "section", "time_period"
    If a value is not found, return null for that key.

    Example 1:
    Query: "Pull the risk factors from Apple's latest 10-K."
    JSON: {{"ticker": "AAPL", "company_name": "Apple", "filing_type": "10-K", "section": "part1item1a", "time_period": "1"}}

    Example 2:
    Query: "Get me the latest news for Microsoft for the last 5 years."
    JSON: {{"ticker": "MSFT", "company_name": "Microsoft", "filing_type": null, "section": null, "time_period": "5"}}

    Example 3:
    Query: "Show me a chart"
    JSON: {{"ticker": null, "company_name": null, "filing_type": null, "section": null, "time_period": "1"}}

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

        # If no new entities found, check if we can use previous context
        if not ticker and not company_name:
            # Check if we have previous context
            prev_ticker = state.get("ticker")
            prev_company = state.get("company_name")
            
            if prev_ticker or prev_company:
                print(f"No new company found, using previous context: {prev_company or prev_ticker}")
                # Don't return error, just use None values
                # The merge_entities will preserve previous values
                ticker = None
                company_name = None
            else:
                msg = "I couldn't identify the company in your query. Please be more specific."
                print("No ticker or company name found and no previous context.")
                return {
                    "final_answer": msg,
                    "messages": [AIMessage(content=msg)]
                }

        print(f"Entities extracted: Ticker={ticker}, Company={company_name}, Filing={filing_type}, Section={section}, Time={time_period}, Chart={create_chart}")
        
        # Map the section code
        mapped_section = map_section_code(section) if section else None
        
        result = {
            "ticker": ticker, 
            "company_name": company_name,
            "filing_type": filing_type,
            "section": mapped_section,
            "time_period": time_period,
            "create_chart": create_chart  # ADD THIS
        }
        
        # Remove None values so merge_entities doesn't overwrite
        result = {k: v for k, v in result.items() if v is not None}
        
        return result
        
    except Exception as e:
        error_message = f"Failed to extract entities. Error: {e}"
        print(error_message)
        return {
            "final_answer": error_message,
            "messages": [AIMessage(content=error_message)]
        }