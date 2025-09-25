# workflows/get_stock_data_and_chart.py (Recommended File Name)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from state import FinanceAgentState
from io import StringIO 



def get_stock_data_and_chart_node(state: FinanceAgentState) -> dict:
    """
    Fetches stock data, updates the state with metrics, and conditionally
    generates a price chart based on the 'create_chart' flag in the state.
    """
    # --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
    if state.get("final_answer"):
        print("\n" + state["final_answer"])
        print("-" * 30)
    # --------------------------------------------------
    print("---NODE: Getting Data & Conditionally Creating Chart---")
    
    ticker = state.get("ticker")
    # Retrieve the new boolean flag
    request_chart = state.get("create_chart", False) 
    
    if not ticker:
        error_message = "No stock ticker found in the state. Cannot fetch data."
        print(f"Error: {error_message}")
        return {"final_answer": error_message}
        
    try:
        # --- 1. DATA FETCHING (Unified Action) ---
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        
        # Prepare the necessary data formats
        price_json_string = hist.to_json(orient='split')
        
        # Prepare the summary metrics dictionary (for structured_data key)
        summary_metrics = {
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }
        
        # --- 2. CONDITIONAL CHART GENERATION ---
        chart_message = ""
        if request_chart:
            company_name = state.get("company_name", ticker)
            
            # Use data fetched above
            price_history_df = pd.read_json(StringIO(price_json_string), orient='split')
            price_history_df.index = pd.to_datetime(price_history_df.index)

            # Create 'reports' directory if it doesn't exist
            os.makedirs('./charts', exist_ok=True)
            chart_path = f'./charts/{company_name}_price_chart.png'

            plt.figure(figsize=(10, 5))
            plt.plot(price_history_df.index, price_history_df['Close'])
            plt.title(f'{company_name} Stock Price (Last Year)')
            plt.xlabel('Date')
            plt.ylabel('Closing Price (USD)')
            plt.grid(True)
            plt.savefig(chart_path)
            plt.close()
            
            chart_message = f"Chart saved successfully to {chart_path}."
        
        # --- 3. FINAL STATE RETURN ---
        data_fetch_message = f"Successfully fetched metrics for {ticker}."
        
        return {
            "price_history_json": price_json_string, # Raw data for report/chart node
            "structured_data": summary_metrics,      # Summary metrics for report/display
            "final_answer": f"{data_fetch_message} {chart_message}".strip(),
            "create_chart": request_chart # Preserve the flag in state
        }
    
    except Exception as e:
        error_message = f"Failed to fetch stock data or create chart for {ticker}. Error: {e}"
        print(f"Error: {error_message}")
        return {"final_answer": error_message}