from fastapi import FastAPI, HTTPException
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from state import FinanceAgentState
from workflows.get_financial_news import get_financial_news_node
from workflows.extract_entities import extract_entities_node
from workflows.get_sec_filing_section import get_sec_filing_section_node
from workflows.get_stock_data_and_chart import get_stock_data_and_chart_node
from workflows.curate_report import curate_report_node

app = FastAPI(
    title="Financial Analyst API",
    description="API for financial analysis including stock data, SEC filings, and news",
    version="1.0.0"
)

class AnalysisRequest(BaseModel):
    query: str
    create_chart: bool = False

class StockDataRequest(BaseModel):
    company_name: str
    ticker: Optional[str] = None
    time_period: Optional[str] = "1"
    create_chart: bool = False

class SECFilingRequest(BaseModel):
    company_name: str
    ticker: Optional[str] = None
    filing_type: str = "10-K"
    section: str = "1A"

class NewsRequest(BaseModel):
    company_name: str
    ticker: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Financial Analyst API"}

@app.post("/analyze")
async def analyze(request: AnalysisRequest) -> Dict[str, Any]:
    try:
        
        state = FinanceAgentState()
        state["user_query"] = request.query
        state["create_chart"] = request.create_chart

        entities_result = extract_entities_node(state)
        if "final_answer" in entities_result:
            return {"error": entities_result["final_answer"]}
        
        state.update(entities_result)

        stock_result = get_stock_data_and_chart_node(state)
        if "final_answer" in stock_result and "error" in stock_result["final_answer"].lower():
            return {"error": stock_result["final_answer"]}
        state.update(stock_result)

        sec_result="dummy"

      
        news_result = get_financial_news_node(state)
        if "final_answer" in news_result and "error" in news_result["final_answer"].lower():
            return {"error": news_result["final_answer"]}
        state.update(news_result)

       
        report_result = curate_report_node(state)
        
        return {
            "report": report_result.get("final_answer"),
            "stock_data": stock_result.get("structured_data"),
            "news": news_result.get("news_results"),
            "chart_path": stock_result.get("chart_path") if request.create_chart else None,
            "filing_info": sec_result.get("filing_info")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stock-data")
async def get_stock_data(request: StockDataRequest) -> Dict[str, Any]:
    try:
        state = FinanceAgentState()
        state["company_name"] = request.company_name
        state["ticker"] = request.ticker
        state["time_period"] = request.time_period
        state["create_chart"] = request.create_chart

        result = get_stock_data_and_chart_node(state)
        if "final_answer" in result and "error" in result["final_answer"].lower():
            return {"error": result["final_answer"]}

        return {
            "data": result.get("structured_data"),
            "chart_path": result.get("chart_path") if request.create_chart else None,
            "data_range": result.get("data_range")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sec-filing")
async def get_sec_filing(request: SECFilingRequest) -> Dict[str, Any]:
    try:
       
        state = FinanceAgentState()
        if getattr(request, "company_name", None):
            state["company_name"] = request.company_name
        if getattr(request, "ticker", None):
            state["ticker"] = request.ticker
       
        state["filing_type"] = request.filing_type or "10-K"
       
        state["section"] = request.section or "1A"

       
        result = await asyncio.to_thread(get_sec_filing_section_node, state)

        
        def _extract_messages(msgs: Any) -> List[str]:
            out: List[str] = []
            if not msgs:
                return out
            for m in msgs:
                try:
                    
                    out.append(m.content if hasattr(m, "content") else str(m))
                except Exception:
                    out.append(str(m))
            return out

        messages = _extract_messages(result.get("messages"))

       
        tool_result = result.get("tool_result") or result.get("content") or None
        final_answer = result.get("final_answer") or ""

        if tool_result:
            return {
                "success": True,
                "filing_info": result.get("filing_info", {}),
                "content": tool_result,
                "message": final_answer,
                "messages": messages,
            }

        
        error_detail = final_answer or "Extraction failed or returned no content."
        raise HTTPException(status_code=404, detail=error_detail)

    except HTTPException:
       
        raise
    except Exception as e:
       
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/news")
async def get_news(request: NewsRequest) -> Dict[str, Any]:
    try:
        state = FinanceAgentState()
        state["company_name"] = request.company_name
        state["ticker"] = request.ticker

        result = get_financial_news_node(state)
        if "final_answer" in result and "error" in result["final_answer"].lower():
            return {"error": result["final_answer"]}

        return {
            "news": result.get("news_results")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))