# workflows/extract_entities.py

import json
from langchain_ollama import OllamaLLM
from state import FinanceAgentState
from rich.markdown import Markdown as RichMarkdown
from langchain_core.messages import AIMessage, HumanMessage 

llm = OllamaLLM(model="llama3.2", format="json")

def map_section_code(section):
    section_mapping = {
        "1A": ["part2item1a", "part1item1"],  # Try Risk Factors section first, then Business section
        "7": "part2item7",      # Management's Discussion and Analysis
        "1": "part1item1",      # Business
        "2": "part2item2",      # Financial Information
        # Add more mappings as needed
    }
    result = section_mapping.get(section, section)
    # If we got a list of alternatives, return the first one
    return result[0] if isinstance(result, list) else result

def extract_entities_node(state: FinanceAgentState) -> dict:
    print("---NODE: Extracting Entities---")
    user_query = state.get("user_query", "")
    
    prompt = f"""Extract financial entities from the user query and return ONLY a JSON object.

Extract these fields:
- ticker: Stock symbol (e.g., AAPL, MSFT, NVDA)
- company_name: Company name (e.g., Apple, Microsoft, NVIDIA)
- filing_type: Either "10-K" or "10-Q" or null
- section: Either "part2item1a" (risks), "part2item7" (MD&A), "part1item1" (business), "part2item2" (financials), or null
- time_period: Number of years (default "1")

Examples:

Query: "Pull the risk factors from Apple's latest 10-K."
{{"ticker": "AAPL", "company_name": "Apple", "filing_type": "10-K", "section": "part2item1a", "time_period": "1"}}

Query: "Get me the latest news for Microsoft."
{{"ticker": "MSFT", "company_name": "Microsoft", "filing_type": null, "section": null, "time_period": "1"}}

Query: "What is the current P/E ratio for NVIDIA (NVDA)?"
{{"ticker": "NVDA", "company_name": "NVIDIA", "filing_type": null, "section": null, "time_period": "1"}}

Query: "Show me Tesla stock chart"
{{"ticker": "TSLA", "company_name": "Tesla", "filing_type": null, "section": null, "time_period": "1"}}

Now extract from this query:
Query: "{user_query}"

Return ONLY the JSON object, nothing else:"""
    
    try:
        response_str = llm.invoke(prompt)
        
        # Clean up response - remove markdown code blocks if present
        response_str = response_str.strip()
        if response_str.startswith("```json"):
            response_str = response_str[7:]
        if response_str.startswith("```"):
            response_str = response_str[3:]
        if response_str.endswith("```"):
            response_str = response_str[:-3]
        response_str = response_str.strip()
        
        # Debug: print raw response
        print(f"LLM Response: {response_str}")
        
        entities = json.loads(response_str)
        
        ticker = entities.get("ticker")
        company_name = entities.get("company_name")
        filing_type = entities.get("filing_type")
        section = entities.get("section")
        time_period = entities.get("time_period", "1")  # Default to "1" if not provided

        if not ticker and not company_name:
            msg = "I couldn't identify the company in your query. Please be more specific."
            print("No ticker or company name found.")
            print(RichMarkdown(msg))
            return {
                "ticker": None,
                "company_name": None,
                "filing_type": None,
                "section": None,
                "time_period": "1",
                "final_answer": msg,
                "messages": [
                    AIMessage(content=msg)
                ]
            }

        print(f"✓ Entities extracted: Ticker={ticker}, Company={company_name}, Filing={filing_type}, Section={section}, Time Period={time_period}")
        
        # Map the section code to its proper SEC API identifier
        mapped_section = map_section_code(section) if section else None
        
        return {
            "ticker": ticker, 
            "company_name": company_name,
            "filing_type": filing_type,
            "section": mapped_section,
            "time_period": time_period,
            "messages": [
                AIMessage(content=f"Got it - you're asking about {company_name or ticker}. I'll look for {filing_type or 'relevant information'} in the {section or 'main'} section for {time_period or '1'} year(s).")
            ]
        }
        
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse LLM response. Error: {e}. Response was: {response_str}"
        print(f"✗ {error_message}")
        return {
            "ticker": None,
            "company_name": None,
            "filing_type": None,
            "section": None,
            "time_period": "1",
            "final_answer": "I had trouble understanding that query. Please try rephrasing.",
            "messages": [AIMessage(content="I had trouble understanding that query. Please try rephrasing.")]
        }
    except Exception as e:
        error_message = f"Failed to extract entities. Error: {e}"
        print(f"✗ {error_message}")
        return {
            "ticker": None,
            "company_name": None,
            "filing_type": None,
            "section": None,
            "time_period": "1",
            "final_answer": error_message,
            "messages": [AIMessage(content=error_message)]
        }