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

def _normalize_to_item_code(raw: str) -> str:
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

def _download_html(url: str) -> str:
    headers = {"User-Agent": "FinSight Agent (contact: pranaybhagwat04@gmail.com)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


def _fallback_extract_from_html(html: str, item: str) -> str:
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


def _print_section_to_terminal(ticker: str, filing_type: str, section_label: str, text: str, filing_url: str = None, filing_date: str = None):
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
    syntax = Syntax(display_text, "text", theme="monokai", line_numbers=False)
    console.print(syntax)

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

        console.print("[yellow]Extractor failed → switching to fallback HTML parsing...[/yellow]")
        html = _download_html(filing_url)

        fallback_key = item_code if item_code in SECTION_PATTERNS else "1A"
        extracted = _fallback_extract_from_html(html, fallback_key)

        if not extracted:
            extracted = _fallback_extract_from_html(html, "1")
            fallback_key = "1" if extracted else fallback_key

        if extracted:
            msg = f"(Fallback) Extracted section {fallback_key} for {ticker}."
            console.print(Panel(msg, style="green"))
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

        msg = f"Extractor API + fallback failed. Error={extractor_error}"
        console.print(Panel(msg, style="red"))
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    except Exception as e:
        err = f"General extraction failure: {e}"
        console.print(Panel(err, style="red"))
        return {"final_answer": err, "messages": [AIMessage(content=err)]}

