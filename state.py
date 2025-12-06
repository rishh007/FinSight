# state.py

from typing_extensions import TypedDict
from typing import Annotated, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class FinanceAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # for conversation history 
    user_query: str
    intent: str
    should_continue: bool
    create_chart : bool 

    time_period: str  

# tool inputs 
    company_name: Optional[str]
    ticker: Optional[str]
    filing_type: Optional[str]
    section: Optional[str]
    
# tool outputs
    tool_result: Optional[str]
    structured_data: Optional[dict]   
    price_history_json : Optional[str] 
    news_results: Optional[str]  # get_financial_news updates this 

# final output 
    final_answer: Optional[str]
    report_data: Optional[dict]  # get_report

    rag_answer: Optional[str]

