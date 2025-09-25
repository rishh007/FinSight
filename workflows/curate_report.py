# # get_report.py (Corrected Logic)

# import os
# import markdown
# from state import FinanceAgentState # Corrected import path

# # Assuming your imports are all correct...

# def curate_report_node(state: FinanceAgentState) -> dict:
#     """
#     Synthesizes information from multiple sources to generate a comprehensive
#     financial analyst report and returns it as a Markdown string.
#     """

#     # --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
#     if state.get("final_answer"):
#         print("\n" + state["final_answer"])
#         print("-" * 30)
#     # -------------------------------------------------- 
    
#     print("---NODE: Generating Analyst Report---")

#     # --- 1. RETRIEVE DATA FROM CORRECT CHANNELS ---
#     company_name = state.get("company_name", "Unknown Company")
    
#     # Pulling data from the correct primary channels:
#     risk_summary = state.get("tool_result", "No key risks found in recent filings (SEC Filing).") # Changed from risk_summary
#     key_metrics = state.get("structured_data", {}) # Changed from stock_performance_summary
#     news_articles = state.get("news_results", []) # Changed from news_summary
    
#     # Format stock performance summary for the report body
#     if key_metrics:
#         stock_performance_text = (
#             f"Current Price: **{key_metrics.get('current_price', 'N/A')}** | "
#             f"Market Cap: **{key_metrics.get('market_cap', 'N/A')}** | "
#             f"P/E Ratio: **{key_metrics.get('pe_ratio', 'N/A')}**"
#         )
#     else:
#         stock_performance_text = "Stock performance data is unavailable."


#     # Check if a company name was successfully extracted
#     if company_name == "Unknown Company":
#         return {"final_answer": "Failed to generate a report. Could not determine the company from your query."}

#     # --- 2. FORMAT NEWS SUMMARY ---
#     if news_articles and isinstance(news_articles, list):
#         news_summary_text = "\n".join([f"- **{article.get('title', 'N/A')}**" for article in news_articles])
#     else:
#         # Simulate data only if absolutely necessary (e.g., if news node failed)
#         news_summary_text = (
#             f"- **{company_name}** Announces Strategic Update (Simulated)\n"
#             f"- Analyst upgrades **{company_name}** to 'Buy' (Simulated)"
#         )

#     # --- 3. CONSTRUCT REPORT ---
#     report_content_md = f"""
# # Financial Analyst Report for {company_name}

# ## Executive Summary
# This report combines key insights from market data, SEC filings, and news sentiment.

# ## 1. Key Business Risks
# Based on the company's most recent SEC filing:
# {risk_summary}

# ## 2. Recent Market Performance (Summary)
# {stock_performance_text}

# ## 3. Recent News & Catalysts
# An analysis of recent financial news reveals the following events:
# {news_summary_text}

# ---

# **Note:** This report is for informational purposes only.
# """
    

    
#     # --- 4. RETURN FINAL STATE ---
#     print("---NODE: Displaying Report in Markdown---")
#     # html_output = markdown.markdown(report_content_md)
#     # print(html_output)
#     from IPython.display import Markdown, display

#     display(Markdown(report_content_md))
#     os.makedirs('./reports', exist_ok=True)
#     report_path = f'./reports/{company_name}_analyst_report.png'

#     return {
#         # Store a copy of the final data structure for memory/persistence
#         "report_data": {
#             "company_name": company_name,
#             "risks": risk_summary,
#             "performance_metrics": stock_performance_text,
#             "news_count": len(news_articles) if isinstance(news_articles, list) else 0
#         },
#         "final_answer": report_content_md
#     }
# get_report.py (Enhanced with Pretty Print and DOCX - Original State Logic Preserved)
# -----------------
# import os
# import markdown
# from datetime import datetime
# from docx import Document
# from docx.shared import Inches, Pt
# from docx.enum.text import WD_ALIGN_PARAGRAPH
# from rich.console import Console
# from rich.panel import Panel
# from rich.markdown import Markdown as RichMarkdown
# from state import FinanceAgentState  # Corrected import path

# # Initialize Rich console for pretty printing
# console = Console()

# def create_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics):
#     """Create a professional DOCX report"""
#     doc = Document()
    
#     # Set margins
#     sections = doc.sections
#     for section in sections:
#         section.top_margin = Inches(1)
#         section.bottom_margin = Inches(1)
#         section.left_margin = Inches(1.2)
#         section.right_margin = Inches(1.2)
    
#     # Title
#     title = doc.add_heading(f'Financial Analyst Report', 0)
#     title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
#     # Company name
#     company_heading = doc.add_heading(company_name, level=1)
#     company_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
#     # Date
#     date_para = doc.add_paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
#     date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
#     doc.add_paragraph()
    
#     # Executive Summary
#     doc.add_heading('Executive Summary', level=2)
#     doc.add_paragraph(
#         'This report combines key insights from market data, SEC filings, and news sentiment.'
#     )
    
#     # Key Metrics Table (if available)
#     if key_metrics:
#         doc.add_heading('Key Financial Metrics', level=2)
#         table = doc.add_table(rows=1, cols=2)
#         table.style = 'Light Grid Accent 1'
        
#         # Header
#         header_cells = table.rows[0].cells
#         header_cells[0].text = 'Metric'
#         header_cells[1].text = 'Value'
        
#         # Add metrics
#         for key, value in key_metrics.items():
#             if value and value != 'N/A':
#                 row = table.add_row().cells
#                 row[0].text = key.replace('_', ' ').title()
#                 row[1].text = str(value)
        
#         doc.add_paragraph()
    
#     # Business Risks
#     doc.add_heading('Key Business Risks', level=2)
#     doc.add_paragraph(risk_summary)
    
#     # Market Performance
#     doc.add_heading('Market Performance Summary', level=2)
#     doc.add_paragraph(stock_performance_text)
    
#     # News Analysis
#     doc.add_heading('Recent News & Catalysts', level=2)
#     if news_articles and isinstance(news_articles, list):
#         doc.add_paragraph('Recent developments include:')
#         for article in news_articles[:5]:
#             title = article.get('title', 'N/A')
#             doc.add_paragraph(f"• {title}", style='List Bullet')
#     else:
#         doc.add_paragraph('Recent news analysis shows ongoing market developments.')
    
#     # Disclaimer
#     doc.add_paragraph()
#     disclaimer = doc.add_heading('Disclaimer', level=2)
#     doc.add_paragraph(
#         'This report is for informational purposes only and should not be considered '
#         'investment advice. Consult a qualified financial advisor before making decisions.'
#     )
    
#     return doc

# def curate_report_node(state: FinanceAgentState) -> dict:
#     """
#     Synthesizes information from multiple sources to generate a comprehensive
#     financial analyst report and returns it as a Markdown string.
#     """

#     # --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
#     if state.get("final_answer"):
#         print("\n" + state["final_answer"])
#         print("-" * 30)
#     # -------------------------------------------------- 
    
#     # Pretty print node start
#     console.print(Panel("🔄 NODE: Generating Analyst Report", style="bold green"))

#     # --- 1. RETRIEVE DATA FROM CORRECT CHANNELS ---
#     company_name = state.get("company_name", "Unknown Company")
    
#     # Pulling data from the correct primary channels:
#     risk_summary = state.get("tool_result", "No key risks found in recent filings (SEC Filing).") # Changed from risk_summary
#     key_metrics = state.get("structured_data", {}) # Changed from stock_performance_summary
#     news_articles = state.get("news_results", []) # Changed from news_summary
    
#     # Format stock performance summary for the report body
#     if key_metrics:
#         stock_performance_text = (
#             f"Current Price: **{key_metrics.get('current_price', 'N/A')}** | "
#             f"Market Cap: **{key_metrics.get('market_cap', 'N/A')}** | "
#             f"P/E Ratio: **{key_metrics.get('pe_ratio', 'N/A')}**"
#         )
#     else:
#         stock_performance_text = "Stock performance data is unavailable."


#     # Check if a company name was successfully extracted
#     if company_name == "Unknown Company":
#         error_msg = "Failed to generate a report. Could not determine the company from your query."
#         console.print(Panel(error_msg, style="bold red"))
#         return {"final_answer": error_msg}

#     # --- 2. FORMAT NEWS SUMMARY ---
#     if news_articles and isinstance(news_articles, list):
#         news_summary_text = "\n".join([f"- **{article.get('title', 'N/A')}**" for article in news_articles])
#     else:
#         # Simulate data only if absolutely necessary (e.g., if news node failed)
#         news_summary_text = (
#             f"- **{company_name}** Announces Strategic Update (Simulated)\n"
#             f"- Analyst upgrades **{company_name}** to 'Buy' (Simulated)"
#         )

#     # --- 3. CONSTRUCT REPORT ---
#     report_content_md = f"""
# # Financial Analyst Report for {company_name}

# ## Executive Summary
# This report combines key insights from market data, SEC filings, and news sentiment.

# ## 1. Key Business Risks
# Based on the company's most recent SEC filing:
# {risk_summary}

# ## 2. Recent Market Performance (Summary)
# {stock_performance_text}

# ## 3. Recent News & Catalysts
# An analysis of recent financial news reveals the following events:
# {news_summary_text}

# ---

# **Note:** This report is for informational purposes only.
# """
    
#     # Pretty print the report
#     console.print("\n" + "="*60)
#     console.print(RichMarkdown(report_content_md))
#     console.print("="*60)
    
#     # Create and save DOCX
#     try:
#         os.makedirs('./reports', exist_ok=True)
        
#         # Generate DOCX
#         doc = create_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics)
        
#         # Save with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{company_name.replace(' ', '_')}_report_{timestamp}.docx"
#         docx_path = f"./reports/{filename}"
#         doc.save(docx_path)
        
#         console.print(Panel(f"✅ DOCX saved: {docx_path}", style="bold green"))
        
#     except Exception as e:
#         console.print(Panel(f"⚠️  DOCX save failed: {str(e)}", style="bold yellow"))

#     # --- 4. RETURN FINAL STATE ---
#     print("---NODE: Displaying Report in Markdown---")
#     # html_output = markdown.markdown(report_content_md)
#     # print(html_output)
    
#     try:
#         from IPython.display import Markdown, display
#         display(Markdown(report_content_md))
#     except ImportError:
#         pass  # Not in Jupyter environment
    
#     # ORIGINAL STATE LOGIC - UNCHANGED
#     return {
#         # Store a copy of the final data structure for memory/persistence
#         "report_data": {
#             "company_name": company_name,
#             "risks": risk_summary,
#             "performance_metrics": stock_performance_text,
#             "news_count": len(news_articles) if isinstance(news_articles, list) else 0
#         },
#         "final_answer": report_content_md
#     }
# get_report.py (Enhanced with Professional Styling and Price Charts)

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown as RichMarkdown
from state import FinanceAgentState

# Try to import docx with fallback
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.shared import OxmlElement, qn
    DOCX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: python-docx not available: {e}")
    DOCX_AVAILABLE = False

# Initialize Rich console for pretty printing
console = Console()

def create_price_chart(company_name, symbol=None, days=90):
    """Create a professional price chart for the company"""
    try:
        # Try to get symbol from company name if not provided
        if not symbol:
            # Simple mapping for common companies - you can expand this
            symbol_map = {
                'salesforce': 'CRM',
                'apple': 'AAPL',
                'microsoft': 'MSFT',
                'google': 'GOOGL',
                'tesla': 'TSLA',
                'amazon': 'AMZN',
                'meta': 'META',
                'nvidia': 'NVDA'
            }
            symbol = symbol_map.get(company_name.lower().split()[0], company_name.upper()[:4])
        
        # Download stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return None
        
        # Create professional chart
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Plot price with gradient fill
        ax.plot(stock_data.index, stock_data['Close'], 
               color='#1f77b4', linewidth=2.5, alpha=0.9)
        ax.fill_between(stock_data.index, stock_data['Close'], 
                       alpha=0.3, color='#1f77b4')
        
        # Styling
        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Labels and title
        ax.set_title(f'{company_name} Stock Price - Last {days} Days', 
                    fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xlabel('Date', fontsize=12, color='#34495e')
        ax.set_ylabel('Price ($)', fontsize=12, color='#34495e')
        
        # Add current price annotation
        current_price = stock_data['Close'].iloc[-1]
        ax.annotate(f'${current_price:.2f}', 
                   xy=(stock_data.index[-1], current_price),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.8),
                   color='white', fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        # Save chart
        os.makedirs('./reports/charts', exist_ok=True)
        chart_path = f'./reports/charts/{symbol}_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path, current_price, symbol
        
    except Exception as e:
        console.print(Panel(f"⚠️ Chart creation failed: {str(e)}", style="bold yellow"))
        return None, None, None

def create_enhanced_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics):
    """Create a professional DOCX report with enhanced styling"""
    if not DOCX_AVAILABLE:
        return None
    
    doc = Document()
    
    # Set document margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.25)
    
    # Define enhanced styles
    styles = doc.styles
    
    # Custom Title Style
    try:
        title_style = styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    except:
        title_style = styles['Title']
    
    title_font = title_style.font
    title_font.name = 'Segoe UI'
    title_font.size = Pt(28)
    title_font.bold = True
    title_font.color.rgb = RGBColor(31, 78, 121)  # Professional blue
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.space_after = Pt(18)
    
    # Custom Header Style
    try:
        header_style = styles.add_style('CustomHeader', WD_STYLE_TYPE.PARAGRAPH)
    except:
        header_style = styles['Heading 2']
    
    header_font = header_style.font
    header_font.name = 'Segoe UI'
    header_font.size = Pt(16)
    header_font.bold = True
    header_font.color.rgb = RGBColor(52, 73, 94)  # Dark blue-gray
    header_style.paragraph_format.space_before = Pt(16)
    header_style.paragraph_format.space_after = Pt(8)
    
    # Custom Body Style
    try:
        body_style = styles.add_style('CustomBody', WD_STYLE_TYPE.PARAGRAPH)
    except:
        body_style = styles['Normal']
    
    body_font = body_style.font
    body_font.name = 'Segoe UI'
    body_font.size = Pt(11)
    body_font.color.rgb = RGBColor(44, 62, 80)  # Professional dark gray
    body_style.paragraph_format.space_after = Pt(6)
    body_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    
    # Document Header with company name
    title_para = doc.add_paragraph()
    title_run = title_para.add_run(f'Financial Analysis Report')
    title_run.font.name = 'Segoe UI'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(31, 78, 121)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Company name in larger font
    company_para = doc.add_paragraph()
    company_run = company_para.add_run(company_name)
    company_run.font.name = 'Segoe UI'
    company_run.font.size = Pt(20)
    company_run.font.bold = True
    company_run.font.color.rgb = RGBColor(231, 76, 60)  # Accent red
    company_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date with better formatting
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
    date_run.font.name = 'Segoe UI'
    date_run.font.size = Pt(10)
    date_run.font.italic = True
    date_run.font.color.rgb = RGBColor(127, 140, 141)
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()  # Space
    
    # Try to create and insert price chart
    chart_path, current_price, symbol = create_price_chart(company_name)
    if chart_path and os.path.exists(chart_path):
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = chart_para.runs[0] if chart_para.runs else chart_para.add_run()
        run.add_picture(chart_path, width=Inches(6))
        doc.add_paragraph()  # Space after chart
    
    # Executive Summary with better styling
    exec_header = doc.add_paragraph()
    exec_run = exec_header.add_run('📊 Executive Summary')
    exec_run.font.name = 'Segoe UI'
    exec_run.font.size = Pt(16)
    exec_run.font.bold = True
    exec_run.font.color.rgb = RGBColor(52, 73, 94)
    
    exec_body = doc.add_paragraph()
    exec_text = exec_body.add_run(
        'This comprehensive financial analysis synthesizes insights from SEC filings, '
        'real-time market data, and current news sentiment to provide actionable '
        'investment intelligence.'
    )
    exec_text.font.name = 'Segoe UI'
    exec_text.font.size = Pt(11)
    exec_text.font.color.rgb = RGBColor(44, 62, 80)
    
    # Enhanced Key Metrics Table
    if key_metrics:
        metrics_header = doc.add_paragraph()
        metrics_run = metrics_header.add_run('💰 Key Financial Metrics')
        metrics_run.font.name = 'Segoe UI'
        metrics_run.font.size = Pt(16)
        metrics_run.font.bold = True
        metrics_run.font.color.rgb = RGBColor(52, 73, 94)
        
        # Create professional table
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Light Grid Accent 1'
        
        # Style header row
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Metric'
        header_cells[1].text = 'Value'
        
        for cell in header_cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Segoe UI'
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            # Set cell background color
            cell._tc.get_or_add_tcPr().append(
                OxmlElement('w:shd')).set(qn('w:fill'), '1F4E79')
        
        # Add formatted metrics
        metric_labels = {
            'current_price': 'Current Price',
            'market_cap': 'Market Cap',
            'pe_ratio': 'P/E Ratio',
            'eps': 'EPS',
            'week_52_high': '52-Week High',
            'week_52_low': '52-Week Low'
        }
        
        for key, value in key_metrics.items():
            if value and value != 'N/A':
                row = table.add_row().cells
                row[0].text = metric_labels.get(key, key.replace('_', ' ').title())
                
                # Format value based on type
                if key == 'market_cap' and isinstance(value, (int, float)):
                    formatted_value = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
                elif key in ['current_price', 'week_52_high', 'week_52_low'] and isinstance(value, (int, float)):
                    formatted_value = f"${value:.2f}"
                else:
                    formatted_value = str(value)
                
                row[1].text = formatted_value
                
                # Style data cells
                for cell in row:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.name = 'Segoe UI'
                            run.font.size = Pt(10)
        
        doc.add_paragraph()
    
    # Risk Analysis with emoji
    risk_header = doc.add_paragraph()
    risk_run = risk_header.add_run('⚠️ Key Business Risks')
    risk_run.font.name = 'Segoe UI'
    risk_run.font.size = Pt(16)
    risk_run.font.bold = True
    risk_run.font.color.rgb = RGBColor(52, 73, 94)
    
    risk_body = doc.add_paragraph()
    risk_text = risk_body.add_run(risk_summary if risk_summary.strip() else "Risk analysis data unavailable at this time.")
    risk_text.font.name = 'Segoe UI'
    risk_text.font.size = Pt(11)
    risk_text.font.color.rgb = RGBColor(44, 62, 80)
    
    # Market Performance
    perf_header = doc.add_paragraph()
    perf_run = perf_header.add_run('📈 Market Performance Summary')
    perf_run.font.name = 'Segoe UI'
    perf_run.font.size = Pt(16)
    perf_run.font.bold = True
    perf_run.font.color.rgb = RGBColor(52, 73, 94)
    
    perf_body = doc.add_paragraph()
    # Remove markdown formatting for DOCX
    clean_performance_text = stock_performance_text.replace('**', '')
    perf_text = perf_body.add_run(clean_performance_text)
    perf_text.font.name = 'Segoe UI'
    perf_text.font.size = Pt(11)
    perf_text.font.color.rgb = RGBColor(44, 62, 80)
    
    # News Analysis
    news_header = doc.add_paragraph()
    news_run = news_header.add_run('📰 Recent News & Market Catalysts')
    news_run.font.name = 'Segoe UI'
    news_run.font.size = Pt(16)
    news_run.font.bold = True
    news_run.font.color.rgb = RGBColor(52, 73, 94)
    
    if news_articles and isinstance(news_articles, list):
        news_intro = doc.add_paragraph()
        intro_text = news_intro.add_run('Recent market developments include:')
        intro_text.font.name = 'Segoe UI'
        intro_text.font.size = Pt(11)
        intro_text.font.color.rgb = RGBColor(44, 62, 80)
        
        for i, article in enumerate(news_articles[:5], 1):
            news_item = doc.add_paragraph(style='List Bullet')
            title = article.get('title', 'N/A')
            item_text = news_item.add_run(f"{title}")
            item_text.font.name = 'Segoe UI'
            item_text.font.size = Pt(10)
            item_text.font.color.rgb = RGBColor(44, 62, 80)
    else:
        news_body = doc.add_paragraph()
        news_text = news_body.add_run('Recent market developments and analyst coverage continue to evolve. Monitor financial news for latest updates.')
        news_text.font.name = 'Segoe UI'
        news_text.font.size = Pt(11)
        news_text.font.color.rgb = RGBColor(44, 62, 80)
    
    # Professional Disclaimer
    doc.add_paragraph()
    disclaimer_header = doc.add_paragraph()
    disclaimer_run = disclaimer_header.add_run('⚖️ Important Disclaimer')
    disclaimer_run.font.name = 'Segoe UI'
    disclaimer_run.font.size = Pt(14)
    disclaimer_run.font.bold = True
    disclaimer_run.font.color.rgb = RGBColor(192, 57, 43)  # Warning red
    
    disclaimer_body = doc.add_paragraph()
    disclaimer_text = disclaimer_body.add_run(
        'This report is generated for informational and educational purposes only. It should not be '
        'considered as personalized investment advice, a recommendation to buy or sell securities, '
        'or a guarantee of future performance. All investments carry risk of loss. Past performance '
        'does not guarantee future results. Please consult with a qualified financial advisor '
        'before making any investment decisions.'
    )
    disclaimer_text.font.name = 'Segoe UI'
    disclaimer_text.font.size = Pt(9)
    disclaimer_text.font.italic = True
    disclaimer_text.font.color.rgb = RGBColor(127, 140, 141)
    
    return doc

def curate_report_node(state: FinanceAgentState) -> dict:
    """
    Synthesizes information from multiple sources to generate a comprehensive
    financial analyst report and returns it as a Markdown string.
    """

    # --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
    if state.get("final_answer"):
        print("\n" + state["final_answer"])
        print("-" * 30)
    # -------------------------------------------------- 
    
    # Pretty print node start
    console.print(Panel("🔄 NODE: Generating Professional Analyst Report", style="bold green"))

    # --- 1. RETRIEVE DATA FROM CORRECT CHANNELS ---
    company_name = state.get("company_name", "Unknown Company")
    
    # Pulling data from the correct primary channels:
    risk_summary = state.get("tool_result", "No key risks found in recent filings (SEC Filing).") # Changed from risk_summary
    key_metrics = state.get("structured_data", {}) # Changed from stock_performance_summary
    news_articles = state.get("news_results", []) # Changed from news_summary
    
    # Format stock performance summary for the report body
    if key_metrics:
        stock_performance_text = (
            f"Current Price: **${key_metrics.get('current_price', 'N/A')}** | "
            f"Market Cap: **${key_metrics.get('market_cap', 'N/A'):,}** | "
            f"P/E Ratio: **{key_metrics.get('pe_ratio', 'N/A')}**"
        )
    else:
        stock_performance_text = "Stock performance data is unavailable."

    # Check if a company name was successfully extracted
    if company_name == "Unknown Company":
        error_msg = "Failed to generate a report. Could not determine the company from your query."
        console.print(Panel(error_msg, style="bold red"))
        return {"final_answer": error_msg}

    # --- 2. FORMAT NEWS SUMMARY ---
    if news_articles and isinstance(news_articles, list):
        news_summary_text = "\n".join([f"- **{article.get('title', 'N/A')}**" for article in news_articles])
    else:
        # Simulate data only if absolutely necessary (e.g., if news node failed)
        news_summary_text = (
            f"- **{company_name}** market developments ongoing\n"
            f"- Analyst coverage and market sentiment analysis continues"
        )

    # --- 3. CONSTRUCT REPORT ---
    report_content_md = f"""
# 📊 Financial Analyst Report for {company_name}

## Executive Summary
This report combines key insights from market data, SEC filings, and news sentiment.

## 1. Key Business Risks
Based on the company's most recent SEC filing:
{risk_summary}

## 2. Recent Market Performance (Summary)
{stock_performance_text}

## 3. Recent News & Catalysts
An analysis of recent financial news reveals the following events:
{news_summary_text}

---

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
**Note:** This report is for informational purposes only.
"""
    
    # Pretty print the report
    console.print("\n" + "="*60)
    console.print(RichMarkdown(report_content_md))
    console.print("="*60)
    
    # Create and save enhanced DOCX
    if DOCX_AVAILABLE:
        try:
            os.makedirs('./reports', exist_ok=True)
            
            # Generate enhanced DOCX
            doc = create_enhanced_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics)
            
            if doc:  # Only save if doc was created successfully
                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{company_name.replace(' ', '_')}_Professional_Report_{timestamp}.docx"
                docx_path = f"./reports/{filename}"
                doc.save(docx_path)
                
                console.print(Panel(f"✅ Enhanced DOCX saved: {docx_path}", style="bold green"))
            
        except Exception as e:
            console.print(Panel(f"⚠️ DOCX save failed: {str(e)}", style="bold yellow"))
    else:
        console.print(Panel("📄 DOCX export skipped (python-docx not available)", style="bold yellow"))

    # --- 4. RETURN FINAL STATE ---
    print("---NODE: Displaying Report in Markdown---")
    
    try:
        from IPython.display import Markdown, display
        display(Markdown(report_content_md))
    except ImportError:
        pass  # Not in Jupyter environment
    
    # ORIGINAL STATE LOGIC - UNCHANGED
    return {
        # Store a copy of the final data structure for memory/persistence
        "report_data": {
            "company_name": company_name,
            "risks": risk_summary,
            "performance_metrics": stock_performance_text,
            "news_count": len(news_articles) if isinstance(news_articles, list) else 0
        },
        "final_answer": report_content_md
    }