import os
from typing import TypedDict, Optional, List
from sec_api import QueryApi, ExtractorApi
from dotenv import load_dotenv
from state import FinanceAgentState

# Load environment variables from .env file
load_dotenv()

def get_sec_filing_section_node(state: FinanceAgentState) -> dict:
    """
    Extracts a specific section from the most recent SEC filing of a given type 
    for a company using data from the agent's state.
    """
  
    ticker = state.get("ticker")
    filing_type = state.get("filing_type", "10-K")  # default to 10-K
    section = state.get("section", "1A")  # default to Risk Factors
    
    if not ticker:
        company_name = state.get("company_name", "")
        if "microsoft" in company_name.lower():
            ticker = "MSFT"
        elif "apple" in company_name.lower():
            ticker = "AAPL"
        elif "tesla" in company_name.lower():
            ticker = "TSLA"
        elif "amazon" in company_name.lower():
            ticker = "AMZN"
        elif "google" in company_name.lower():
            ticker = "GOOGL"
        elif "meta" in company_name.lower():
            ticker = "META"
        elif "nvidia" in company_name.lower():
            ticker = "NVDA"
        elif "salesforce" in company_name.lower():
            ticker = "CRM"
        else:
            error_message = f"Could not determine ticker symbol for company: {company_name}"
            print(f"Error: {error_message}")
            return {"final_answer": error_message}
    
    print(f"Processing SEC filing for {ticker}, form type: {filing_type}, section: {section}")
    
    # init API clients
    sec_api_key = os.getenv("SEC_API_KEY")
    if not sec_api_key:
        error_message = "SEC_API_KEY environment variable is not set."
        print(f"Error: {error_message}")
        return {"final_answer": error_message}
    
    queryApi = QueryApi(api_key=sec_api_key)
    extractorApi = ExtractorApi(api_key=sec_api_key)
    
    try:
        print(f"Searching for latest {filing_type} filing for {ticker}...")
        
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND formType:{filing_type}"
                }
            },
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = queryApi.get_filings(query)
        
        if not response or 'filings' not in response or len(response['filings']) == 0:
            message = f"No recent {filing_type} found for {ticker}."
            print(message)
            return {"final_answer": message}
        
        # get most recent filing
        filing = response['filings'][0]
        filing_url = filing.get('linkToFilingDetails') or filing.get('linkToHtml')
        
        if not filing_url:
            message = f"Could not find filing URL for {ticker} {filing_type}."
            print(message)
            return {"final_answer": message}
        
        print(f"Found filing URL: {filing_url}")
        print(f"Filing date: {filing.get('filedAt', 'Unknown')}")
        
        # Extract the text of the specified section
        print(f"Extracting section {section}...")
        section_text = extractorApi.get_section(
            filing_url=filing_url, 
            section=section, 
            return_type="text"
        )
        
        if not section_text or len(section_text.strip()) == 0:
            message = f"Section {section} appears to be empty or not found in the {filing_type} for {ticker}."
            print(message)
            return {"final_answer": message}
        
        section_text = section_text.strip()
        truncated_text = section_text[:4000] if len(section_text) > 4000 else section_text
        
        word_count = len(section_text.split())
        
        success_message = f"Successfully extracted section {section} from the latest {filing_type} for {ticker}. Retrieved {word_count} words."
        print(success_message)
        
        return {
            "tool_result": truncated_text,
            "filing_info": {
                "ticker": ticker,
                "filing_type": filing_type,
                "section": section,
                "filing_date": filing.get('filedAt'),
                "word_count": word_count,
                "filing_url": filing_url
            },
            "final_answer": success_message
        }
        
    except Exception as e:
        error_message = f"Could not extract section {section} from {filing_type} for {ticker}. Error: {str(e)}"
        print(f"Error: {error_message}")
        
        try:
            print("Attempting alternative extraction method...")
            
            alt_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"ticker": ticker.upper()}},
                            {"term": {"formType": filing_type}}
                        ]
                    }
                },
                "from": 0,
                "size": 1,
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            
            alt_response = queryApi.get_filings(alt_query)
            
            if alt_response and 'filings' in alt_response and len(alt_response['filings']) > 0:
                alt_filing = alt_response['filings'][0]
                alt_filing_url = alt_filing.get('linkToFilingDetails') or alt_filing.get('linkToHtml')
                
                if alt_filing_url:
                    alt_section_text = extractorApi.get_section(
                        filing_url=alt_filing_url,
                        section=section,
                        return_type="text"
                    )
                    
                    if alt_section_text and len(alt_section_text.strip()) > 0:
                        alt_truncated = alt_section_text[:4000] if len(alt_section_text) > 4000 else alt_section_text
                        alt_success = f"Successfully extracted section {section} using alternative method for {ticker}."
                        print(alt_success)
                        
                        return {
                            "tool_result": alt_truncated.strip(),
                            "final_answer": alt_success
                        }
            
            return {"final_answer": f"All extraction methods failed for {ticker} {filing_type} section {section}. Original error: {str(e)}"}
            
        except Exception as alt_e:
            return {"final_answer": f"Primary and alternative extraction failed. Primary: {str(e)}, Alternative: {str(alt_e)}"}

def get_company_risks_node(state: FinanceAgentState) -> dict:
    """
    Simplified function to get risk factors (Section 1A) from 10-K filings
    """
    print("---NODE: Getting Company Risk Factors---")
    
    updated_state = dict(state)
    updated_state["filing_type"] = "10-K"
    updated_state["section"] = "1A"  
    
    return get_sec_filing_section_node(updated_state)