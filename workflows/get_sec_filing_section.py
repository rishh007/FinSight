# import os
# from typing import TypedDict, Optional, List
# from sec_api import QueryApi, ExtractorApi
# from dotenv import load_dotenv
# from state import FinanceAgentState
# from langchain_core.messages import AIMessage, HumanMessage 

# # Load environment variables from .env file
# load_dotenv()

# def get_sec_filing_section_node(state: FinanceAgentState) -> dict:
#     """
#     Extracts a specific section from the most recent SEC filing of a given type 
#     for a company using data from the agent's state.
#     """
  
#     ticker = state.get("ticker")
#     filing_type = state.get("filing_type", "10-K")  # default to 10-K
    
#     # Map common section codes to SEC API section identifiers
#     section_mapping = {
#         "1A": "part1item1a",  # Risk Factors
#         "7": "part2item7",    # Management's Discussion and Analysis
#         "1": "part1item1",    # Business
#         "2": "part2item2",    # Financial Information
#     }
    
#     raw_section = state.get("section", "1A")  # default to Risk Factors
#     section = section_mapping.get(raw_section, raw_section)  # Use mapped value or original if not found
    
#     if not ticker:
#         company_name = state.get("company_name", "")
#         if "microsoft" in company_name.lower():
#             ticker = "MSFT"
#         elif "apple" in company_name.lower():
#             ticker = "AAPL"
#         elif "tesla" in company_name.lower():
#             ticker = "TSLA"
#         elif "amazon" in company_name.lower():
#             ticker = "AMZN"
#         elif "google" in company_name.lower():
#             ticker = "GOOGL"
#         elif "meta" in company_name.lower():
#             ticker = "META"
#         elif "nvidia" in company_name.lower():
#             ticker = "NVDA"
#         elif "salesforce" in company_name.lower():
#             ticker = "CRM"
#         else:
#             error_message = f"Could not determine ticker symbol for company: {company_name}"
#             print(f"---Non AI-MESSAGE Error: {error_message}")
#             return {"final_answer": error_message,
#                     "messages": [AIMessage(content=error_message)]
#                     }
    
#     print(f"Processing SEC filing for {ticker}, form type: {filing_type}, section: {section}")
    
#     # init API clients
#     sec_api_key = os.getenv("SEC_API_KEY")
#     if not sec_api_key:
#         error_message = "SEC_API_KEY environment variable is not set."
#         print(f"Error: {error_message}")
#         return {"final_answer": error_message,
#                  "messages": [AIMessage(content=error_message)]
#                 }
    
#     queryApi = QueryApi(api_key=sec_api_key)
#     extractorApi = ExtractorApi(api_key=sec_api_key)
    
#     try:
#         print(f"Searching for latest {filing_type} filing for {ticker}...")
        
#         query = {
#             "query": {
#                 "query_string": {
#                     "query": f"ticker:{ticker} AND formType:{filing_type}"
#                 }
#             },
#             "from": "0",
#             "size": "1",
#             "sort": [{"filedAt": {"order": "desc"}}]
#         }
        
#         response = queryApi.get_filings(query)
        
#         if not response or 'filings' not in response or len(response['filings']) == 0:
#             message = f"No recent {filing_type} found for {ticker}."
#             print(message)
#             return {"final_answer": message,
#                      "messages": [AIMessage(content=message)]
#                     }
        
#         # get most recent filing
#         filing = response['filings'][0]
#         filing_url = filing.get('linkToFilingDetails') or filing.get('linkToHtml')
        
#         if not filing_url:
#             message = f"Could not find filing URL for {ticker} {filing_type}."
#             print(message)
#             return {"final_answer": message,
#                      "messages": [AIMessage(content=message)]
#                     }
        
#         print(f"Found filing URL: {filing_url}")
#         print(f"Filing date: {filing.get('filedAt', 'Unknown')}")
        
#         # Extract the text of the specified section
#         print(f"Extracting section {section}...")
#         section_text = extractorApi.get_section(
#             filing_url=filing_url, 
#             section=section, 
#             return_type="text"
#         )
        
#         if not section_text or len(section_text.strip()) == 0:
#             message = f"Section {section} appears to be empty or not found in the {filing_type} for {ticker}."
#             print(message)
#             return {"final_answer": message,
#                      "messages": [AIMessage(content=message)]
#                     }
        
#         section_text = section_text.strip()
#         truncated_text = section_text[:4000] if len(section_text) > 4000 else section_text
        
#         word_count = len(section_text.split())
        
#         success_message = f"Successfully extracted section {section} from the latest {filing_type} for {ticker}. Retrieved {word_count} words."
#         print(success_message)
        
#         return {
#             "tool_result": truncated_text,
#             "filing_info": {
#                 "ticker": ticker,
#                 "filing_type": filing_type,
#                 "section": section,
#                 "filing_date": filing.get('filedAt'),
#                 "word_count": word_count,
#                 "filing_url": filing_url
#             },
#             "final_answer": success_message,
#             "messages":[AIMessage(content=f"I’ve extracted the {section} section from the {filing_type} filing for {ticker}. The filing date is {filing.get('filedAt')}.")]
#         }
        
#     except Exception as e:
#         error_message = f"Could not extract section {section} from {filing_type} for {ticker}. Error: {str(e)}"
#         print(f"Error: {error_message}")
        
#         try:
#             print("Attempting alternative extraction method...")
            
#             alt_query = {
#                 "query": {
#                     "bool": {
#                         "must": [
#                             {"term": {"ticker": ticker.upper()}},
#                             {"term": {"formType": filing_type}}
#                         ]
#                     }
#                 },
#                 "from": 0,
#                 "size": 1,
#                 "sort": [{"filedAt": {"order": "desc"}}]
#             }
            
#             alt_response = queryApi.get_filings(alt_query)
            
#             if alt_response and 'filings' in alt_response and len(alt_response['filings']) > 0:
#                 alt_filing = alt_response['filings'][0]
#                 alt_filing_url = alt_filing.get('linkToFilingDetails') or alt_filing.get('linkToHtml')
                
#                 if alt_filing_url:
#                     alt_section_text = extractorApi.get_section(
#                         filing_url=alt_filing_url,
#                         section=section,
#                         return_type="text"
#                     )
                    
#                     if alt_section_text and len(alt_section_text.strip()) > 0:
#                         alt_truncated = alt_section_text[:4000] if len(alt_section_text) > 4000 else alt_section_text
#                         alt_success = f"Successfully extracted section {section} using alternative method for {ticker}."
#                         print(alt_success)
                        
#                         return {
#                             "tool_result": alt_truncated.strip(),
#                             "final_answer": alt_success,
#                             "messages": [AIMessage(content=alt_success)]
#                         }
            
#             return {"final_answer": f"All extraction methods failed for {ticker} {filing_type} section {section}. Original error: {str(e)}", 
#                     "messages": [AIMessage(content=f"AI-Message: All extraction methods failed - {str(e)}")]
#                     }
            
#         except Exception as alt_e:
#             return {"final_answer": f"Primary and alternative extraction failed. Primary: {str(e)}, Alternative: {str(alt_e)}", 
#                      "messages": [AIMessage(content=f"Extraction failed — {str(e)}")]
                    
#                     } 


# def get_company_risks_node(state: FinanceAgentState) -> dict:
#     """
#     Function to get risk factors from 10-K filings.
#     Tries both part2item1a (Risk Factors section) and part1item1 (Business section) as fallback.
#     """
#     print("---NODE: Getting Company Risk Factors---")
    
#     updated_state = dict(state)
#     updated_state["filing_type"] = "10-K"
    
#     # First try part2item1a (dedicated Risk Factors section)
#     updated_state["section"] = "part2item1a"
#     result = get_sec_filing_section_node(updated_state)
    
#     # If that fails, try part1item1 (Business section which often includes risks)
#     if "error" in result.get("final_answer", "").lower():
#         print("---Attempting fallback to Business section (part1item1)---")
#         updated_state["section"] = "part1item1"
#         result = get_sec_filing_section_node(updated_state)
    
#     return result

import os
import re
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi
from langchain_core.messages import AIMessage
from state import FinanceAgentState
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

load_dotenv()

console = Console()
# SEC API Supported Item Codes
SUPPORTED_ITEMS = {
    "1","1A","1B","1C","2","3","4","5","6","7","7A","8",
    "9","9A","9B","9C","10","11","12","13","14","15"
}

# Patterns for HTML fallback extraction
SECTION_PATTERNS = {
    "1A": [r"item\s*1a", r"risk\s*factors"],
    "1":  [r"item\s*1\b", r"business"],
    "2":  [r"item\s*2\b", r"financial"],
    "7":  [r"item\s*7\b", r"management", r"md&a"],
}


# ----------------------------------------------------------
#  NORMALIZE RAW SECTION INPUT → SEC SUPPORTED ITEM CODE
# ----------------------------------------------------------

def _normalize_to_item_code(raw: str) -> str:
    """Convert inputs like 'part2item1a', 'item 1a', '1a' → '1A'."""
    if not raw:
        return ""

    rs = str(raw).strip().lower()

    # "1a", "1", "7"
    m_direct = re.match(r"^(\d{1,2})([a-z]?)$", rs.replace(" ", ""))
    if m_direct:
        num = m_direct.group(1)
        letter = m_direct.group(2).upper() if m_direct.group(2) else ""
        code = f"{num}{letter}"
        if code in SUPPORTED_ITEMS:
            return code

    # "item 1a"
    m = re.search(r"item[\s\-]*([0-9]{1,2})([a-z]?)", rs)
    if m:
        num = m.group(1)
        letter = m.group(2).upper() if m.group(2) else ""
        code = f"{num}{letter}"
        if code in SUPPORTED_ITEMS:
            return code

    # "part2item1a"
    m2 = re.search(r"item(\d{1,2})([a-z]?)", rs)
    if m2:
        num = m2.group(1)
        letter = m2.group(2).upper() if m2.group(2) else ""
        code = f"{num}{letter}"
        if code in SUPPORTED_ITEMS:
            return code

    # default fallback for "risk factors"
    if "risk" in rs:
        return "1A"

    return ""


# ----------------------------------------------------------
#  DOWNLOAD FILING HTML
# ----------------------------------------------------------

def _download_html(url: str) -> str:
    headers = {"User-Agent": "FinSight Agent (contact: pranaybhagwat04@gmail.com)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


# ----------------------------------------------------------
#  FALLBACK HTML PARSER
# ----------------------------------------------------------

def _fallback_extract_from_html(html: str, item: str) -> str:
    """Extract section using BeautifulSoup if Extractor API fails."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    lines = text.split("\n")

    patterns = SECTION_PATTERNS.get(item, [])
    if not patterns:
        patterns = [r"item\s*1a", r"risk\s*factors"]

    regex = re.compile("|".join(patterns), flags=re.IGNORECASE)

    start = None
    for i, line in enumerate(lines):
        if regex.search(line):
            start = i
            break

    if start is None:
        return ""

    section_text = "\n".join(lines[start:start + 300])
    return section_text[:4000]


# ----------------------------------------------------------
#  UTIL: pretty-print the extracted section to terminal
# ----------------------------------------------------------

def _print_section_to_terminal(ticker: str, filing_type: str, section_label: str, text: str, filing_url: str = None, filing_date: str = None):
    """
    Prints the section text (truncated to 4000 chars) to terminal using rich.
    Includes a header panel with metadata and a syntax-highlighted block for readability.
    """
    header = f"{ticker} — {filing_type} — Item {section_label}"
    meta_lines = []
    if filing_date:
        meta_lines.append(f"Filing Date: {filing_date}")
    if filing_url:
        meta_lines.append(f"Filing URL: {filing_url}")
    meta = " | ".join(meta_lines) if meta_lines else ""
    console.print(Panel(Text(header + ("\n" + meta if meta else ""), justify="left"), style="bold cyan"))

    # Ensure text is not too long for terminal — already truncated upstream but double-check
    display_text = text if len(text) <= 4000 else text[:4000] + "\n\n[truncated]"
    # Use Syntax to render plain text nicely
    syntax = Syntax(display_text, "text", theme="monokai", line_numbers=False)
    console.print(syntax)


# ----------------------------------------------------------
#  MAIN NODE
# ----------------------------------------------------------

def get_sec_filing_section_node(state: FinanceAgentState) -> dict:

    ticker = state.get("ticker")
    filing_type = state.get("filing_type", "10-K")
    raw_section = state.get("section", "1A")

    item_code = _normalize_to_item_code(raw_section)
    tried_items = []

    if not ticker:
        msg = "Ticker not provided."
        console.print(Panel(msg, style="red"))
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    api_key = os.getenv("SEC_API_KEY")
    if not api_key:
        msg = "SEC_API_KEY missing."
        console.print(Panel(msg, style="red"))
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    query_api = QueryApi(api_key=api_key)
    extractor_api = ExtractorApi(api_key=api_key)

    try:
        # ----------------------------------------
        # 1) GET LATEST FILING
        # ----------------------------------------
        query = {
            "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"{filing_type}\""}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}],
        }

        resp = query_api.get_filings(query)
        if not resp.get("filings"):
            msg = f"No {filing_type} filing found for {ticker}."
            console.print(Panel(msg, style="red"))
            return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

        filing = resp["filings"][0]
        filing_url = filing.get("linkToFilingDetails")
        filing_date = filing.get("filedAt")

        # ----------------------------------------
        # 2) TRY EXTRACTOR API
        # ----------------------------------------

        extractor_error = None

        if item_code:
            tried_items.append(item_code)
            try:
                console.print(f"[bold yellow]Calling ExtractorApi.get_section → item='{item_code}'[/bold yellow]")

                section_text = extractor_api.get_section(
                    filing_url=filing_url,
                    section=item_code,
                    return_type="text"
                )

                if section_text and section_text.strip():
                    truncated = section_text.strip()[:4000]
                    word_count = len(section_text.split())

                    msg = f"Extracted item {item_code} from {ticker}."
                    console.print(Panel(msg, style="green"))

                    # print the full (truncated to 4k) section to terminal
                    _print_section_to_terminal(ticker, filing_type, item_code, truncated, filing_url, filing_date)

                    return {
                        "tool_result": truncated,
                        "filing_info": {
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "section": item_code,
                            "filing_date": filing_date,
                            "filing_url": filing_url,
                            "tried_items": tried_items
                        },
                        "final_answer": msg,
                        "messages": [AIMessage(content=msg)]
                    }

                else:
                    console.print("[yellow]Extractor API returned empty text for requested section.[/yellow]")

            except Exception as ex:
                extractor_error = str(ex)
                console.print(Panel(f"Extractor API FAILED: {extractor_error}", style="red"))

                # Try to introspect and print response body if present
                try:
                    # sec_api may raise exceptions that include response details in args
                    if hasattr(ex, "response") and ex.response is not None:
                        try:
                            body = ex.response.text
                            body_json = json.loads(body)
                            console.print(Panel(Text(json.dumps(body_json, indent=2)[:2000]), style="red"))
                        except Exception:
                            console.print(Panel(Text(str(ex.response.text)[:2000]), style="red"))
                    else:
                        # try parse from exception message if it contains JSON
                        maybe_json = re.search(r"(\{.*\})", str(ex))
                        if maybe_json:
                            try:
                                parsed = json.loads(maybe_json.group(1))
                                console.print(Panel(Text(json.dumps(parsed, indent=2)[:2000]), style="red"))
                            except Exception:
                                pass
                except Exception as introspect_ex:
                    console.print(Panel(f"Could not introspect API exception: {introspect_ex}", style="red"))

                # Try alternate: if user asked 1 → also try 1A, or vice-versa
                alternates = []
                if item_code.endswith("A"):
                    alternates.append(item_code[:-1])  # "1A" → "1"
                else:
                    alt = item_code + "A"
                    if alt in SUPPORTED_ITEMS:
                        alternates.append(alt)

                # Always try the two common ones
                if "1A" not in alternates: alternates.append("1A")
                if "1" not in alternates: alternates.append("1")

                # Try alternates
                for alt in alternates:
                    if alt in tried_items:
                        continue
                    tried_items.append(alt)
                    try:
                        console.print(f"[bold yellow]ExtractorApi ALT try → '{alt}'[/bold yellow]")
                        section_text = extractor_api.get_section(
                            filing_url=filing_url,
                            section=alt,
                            return_type="text"
                        )
                        if section_text and section_text.strip():
                            truncated = section_text.strip()[:4000]
                            msg = f"Extracted item {alt} (alternate) from {ticker}."
                            console.print(Panel(msg, style="green"))
                            _print_section_to_terminal(ticker, filing_type, alt, truncated, filing_url, filing_date)
                            return {
                                "tool_result": truncated,
                                "final_answer": msg,
                                "messages": [AIMessage(content=msg)],
                                "filing_info": {
                                    "ticker": ticker,
                                    "filing_type": filing_type,
                                    "section": alt,
                                    "filing_date": filing_date,
                                    "filing_url": filing_url,
                                    "tried_items": tried_items
                                }
                            }
                    except Exception as ex2:
                        console.print(Panel(f"Alternate {alt} failed: {ex2}", style="red"))

        # ----------------------------------------
        # 3) FALLBACK TO HTML PARSING
        # ----------------------------------------

        console.print("[yellow]Extractor failed → switching to fallback HTML parsing...[/yellow]")
        html = _download_html(filing_url)

        fallback_key = item_code if item_code in SECTION_PATTERNS else "1A"
        extracted = _fallback_extract_from_html(html, fallback_key)

        if not extracted:
            # Try business as very last resort
            extracted = _fallback_extract_from_html(html, "1")
            fallback_key = "1" if extracted else fallback_key

        if extracted:
            msg = f"(Fallback) Extracted section {fallback_key} for {ticker}."
            console.print(Panel(msg, style="green"))
            # print the extracted section to terminal
            _print_section_to_terminal(ticker, filing_type, fallback_key, extracted, filing_url, filing_date)
            return {
                "tool_result": extracted,
                "final_answer": msg,
                "messages": [AIMessage(content=msg)],
                "filing_info": {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "section": fallback_key,
                    "filing_date": filing_date,
                    "filing_url": filing_url,
                    "tried_items": tried_items
                }
            }

        # FAIL
        msg = f"Extractor API + fallback failed. Error={extractor_error}"
        console.print(Panel(msg, style="red"))
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    except Exception as e:
        err = f"General extraction failure: {e}"
        console.print(Panel(err, style="red"))
        return {"final_answer": err, "messages": [AIMessage(content=err)]}

