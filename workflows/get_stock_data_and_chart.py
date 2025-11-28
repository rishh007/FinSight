import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from state import FinanceAgentState
from io import StringIO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import re
from langchain_core.messages import AIMessage, HumanMessage 

# Initialize Rich console
console = Console()

def get_stock_data_and_chart_node(state: FinanceAgentState) -> dict:
    """
    Fetches stock data, updates the state with metrics, and conditionally
    generates a price chart based on the 'create_chart' flag in the state.
    """
    console.print(Panel("Getting Stock Data & Creating Chart", style="bold green"))

    # --- helper to safely read from state (dict-like or object with get) ---
    def safe_get(obj, key, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, "get"):
                return obj.get(key, default)
            # fallback: attribute access
            return getattr(obj, key, default)
        except Exception:
            return default

    # ---- canonical initialization for company_name and ticker ----
    company_name = safe_get(state, "company_name", None)
    ticker = safe_get(state, "ticker", None)
    request_chart = safe_get(state, "create_chart", False)

    # fallback
    if not company_name:
        company_name = ticker or "Unknown Company"
    company_name = str(company_name).strip() if company_name is not None else "Unknown Company"

    console.print(f"📌 Debug: company_name resolved to '{company_name}', ticker='{ticker}', create_chart={request_chart}'")

    # time_period normalization
    time_period = safe_get(state, "time_period", "1")
    if isinstance(time_period, (int, float)):
        time_period = str(int(time_period))
    elif isinstance(time_period, str):
        match = re.search(r'\d+', time_period)
        time_period = match.group() if match else "1"
    else:
        time_period = "1"

    try:
        period_int = int(time_period)
        if period_int > 10:
            console.print(Panel(f"Warning: Limiting time period from {period_int} to 10 years (yfinance limit)", style="bold yellow"))
            time_period = "10"
        elif period_int < 1:
            console.print(Panel(f"Warning: Adjusting time period from {period_int} to 1 year (minimum)", style="bold yellow"))
            time_period = "1"
    except ValueError:
        console.print(Panel("Warning: Invalid time period, defaulting to 1 year", style="bold yellow"))
        time_period = "1"

    if not ticker:
        error_message = "No stock ticker found in the state. Cannot fetch data."
        console.print(Panel(f"Error: {error_message}", style="bold red"))
        return {"final_answer": error_message, "messages": [AIMessage(content=error_message)]}

    # display params
    params_table = Table(show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")
    params_table.add_row("Ticker", ticker)
    params_table.add_row("Create Chart", "Yes" if request_chart else "No")
    params_table.add_row("Time Period", f"{time_period} year{'s' if int(time_period) != 1 else ''}")
    console.print(params_table)

    try:
        with console.status(f"[bold green]Fetching {time_period} year{'s' if int(time_period) != 1 else ''} of data for {ticker}..."):
            stock = yf.Ticker(ticker)
            if int(time_period) == 1:
                hist = stock.history(period="1y")
            else:
                try:
                    hist = stock.history(period=f"{time_period}y")
                except Exception:
                    console.print("Warning: Using maximum available period", style="bold yellow")
                    hist = stock.history(period="max")
                    if not hist.empty:
                        years_ago = pd.Timestamp.now() - pd.DateOffset(years=int(time_period))
                        hist = hist[hist.index >= years_ago]
            info = {}
            try:
                info = stock.info or {}
            except Exception:
                info = {}

        if hist.empty:
            error_message = f"No historical data available for {ticker} over the requested {time_period} year period."
            console.print(Panel(f"Error: {error_message}", style="bold red"))
            return {"final_answer": error_message, "messages": [AIMessage(content=error_message)]}

        actual_start = hist.index[0].strftime("%Y-%m-%d")
        actual_end = hist.index[-1].strftime("%Y-%m-%d")
        console.print(f"📅 Data retrieved from {actual_start} to {actual_end} ({len(hist)} trading days)")

        price_json_string = hist.to_json(orient='split')

        summary_metrics = {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or (hist['Close'].iloc[-1] if not hist.empty else None),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "52_week_high": info.get("fiftyTwoWeekHigh") or (hist['High'].max() if not hist.empty else None),
            "52_week_low": info.get("fiftyTwoWeekLow") or (hist['Low'].min() if not hist.empty else None),
        }

        # format/display metrics
        metrics_table = Table(title="Key Metrics Retrieved", show_header=True, header_style="bold cyan")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        def format_value(key, value):
            if value is None:
                return "N/A"
            if key == "market_cap" and isinstance(value, (int, float)):
                if value >= 1e12:
                    return f"${value/1e12:.2f}T"
                elif value >= 1e9:
                    return f"${value/1e9:.2f}B"
                else:
                    return f"${value/1e6:.2f}M"
            elif key in ["current_price", "52_week_high", "52_week_low", "eps"]:
                try:
                    return f"${float(value):.2f}"
                except Exception:
                    return str(value)
            elif key == "pe_ratio":
                try:
                    return f"{float(value):.2f}"
                except Exception:
                    return "N/A"
            else:
                return str(value)

        for key, value in summary_metrics.items():
            display_key = key.replace("_", " ").title().replace("52 Week", "52-Week")
            metrics_table.add_row(display_key, format_value(key, value))
        console.print(metrics_table)

        chart_message = ""
        chart_path = None
        if request_chart:
            console.print(f"Creating price chart for {time_period} year period...")
            # use canonical company_name
            safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('./charts', exist_ok=True)
            chart_path = f'./charts/{safe_company_name}_{time_period}yr_chart_{timestamp}.png'

            price_history_df = pd.read_json(StringIO(price_json_string), orient='split')
            price_history_df.index = pd.to_datetime(price_history_df.index)

            plt.figure(figsize=(12, 7))
            plt.subplot(2, 1, 1)
            plt.plot(price_history_df.index, price_history_df['Close'], linewidth=2, alpha=0.8)
            plt.fill_between(price_history_df.index, price_history_df['Close'], alpha=0.3)
            plt.title(f'{company_name} ({ticker}) Stock Price - Last {time_period} Year{"s" if int(time_period) != 1 else ""}',
                      fontsize=14, fontweight='bold')
            plt.ylabel('Closing Price (USD)')
            plt.grid(True, alpha=0.3)

            start_price = price_history_df['Close'].iloc[0]
            end_price = price_history_df['Close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100

            plt.text(0.02, 0.98, f'Start: ${start_price:.2f}', transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.88, f'End: ${end_price:.2f}', transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.78, f'Change: {change_pct:+.1f}%', transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                                                       facecolor='lightgreen' if change_pct >= 0 else 'lightcoral', alpha=0.8))

            plt.subplot(2, 1, 2)
            plt.bar(price_history_df.index, price_history_df['Volume'] / 1e6, alpha=0.7, width=1)
            plt.ylabel('Volume (Millions)')
            plt.xlabel('Date')
            plt.grid(True, alpha=0.3)

            chart_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(
            0.99, -0.02, 
                f"Chart generated on: {chart_timestamp}",
                horizontalalignment='right',
                fontsize=9,
                color='gray'
            )

            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            chart_message = f"Chart saved successfully to {chart_path}."
            console.print(Panel(f"Chart created: {chart_path}", style="bold blue"))

        data_fetch_message = f"Successfully fetched {time_period} year{'s' if int(time_period) != 1 else ''} of data for {ticker}."

        console.print(Panel(f"Success: {data_fetch_message} {chart_message}".strip(), style="bold green"))

        messages = [AIMessage(content=f"Here’s the stock performance for {company_name or ticker} from {actual_start} to {actual_end}. {chart_message}")]
        if chart_path:
            messages.append(AIMessage(content=f"Chart path: {chart_path}"))

        return {
            "price_history_json": price_json_string,
            "structured_data": summary_metrics,
            "time_period": time_period,
            "data_range": f"{actual_start} to {actual_end}",
            "final_answer": f"{data_fetch_message} {chart_message}".strip(),
            "create_chart": request_chart,
            "chart_path": chart_path,
            "messages": messages
        }

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        error_message = f"Failed to fetch stock data or create chart for {ticker} ({time_period} year period). Error: {e}"
        console.print(Panel(f"Error: {error_message}", style="bold red"))
        console.print(traceback_str)
        return {"final_answer": error_message, "messages": [AIMessage(content=error_message)]}
