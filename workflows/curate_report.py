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
from langchain_core.messages import AIMessage, HumanMessage 

# Try to import docx with fallback
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.shared import OxmlElement, qn
    DOCX_AVAILABLE = True
    print("✅ python-docx library loaded successfully")
except ImportError as e:
    print(f"❌ Warning: python-docx not available: {e}")
    DOCX_AVAILABLE = False

# Initialize Rich console for pretty printing
console = Console()

def create_price_chart(company_name, symbol=None, days=90):
    """Create a professional price chart for the company"""
    try:
        # try to get symbol from company name if not provided
        if not symbol:
            symbol_map = {
                'salesforce': 'CRM',
                'apple': 'AAPL',
                'microsoft': 'MSFT',
                'google': 'GOOGL',
                'tesla': 'TSLA',
                'amazon': 'AMZN',
                'meta': 'META',
                'nvidia': 'NVDA',
                'tata motors':'TATMOT', 
            }
            symbol = symbol_map.get(company_name.lower().split()[0], company_name.upper()[:4])
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        console.print(f"📊 Downloading stock data for {symbol}...")
        
        # Fixed: Add auto_adjust parameter and better error handling
        stock_data = yf.download(
            symbol, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False,  
            threads=True
        )
        
        if stock_data.empty:
            console.print(Panel(f"⚠️ No stock data found for {symbol}", style="bold yellow"))
            return None, None, None
        
        # Ensure we have proper data structure - fix for '1-dimensional' error
        if isinstance(stock_data.index, pd.MultiIndex):
            # If MultiIndex (happens with multiple symbols), select first level
            stock_data = stock_data.droplevel(1, axis=1)
        
        # Reset index to ensure proper datetime handling
        stock_data = stock_data.reset_index()
        if 'Date' not in stock_data.columns and stock_data.index.name == 'Date':
            stock_data = stock_data.reset_index()
        
        required_columns = ['Close']
        if not all(col in stock_data.columns for col in required_columns):
            console.print(Panel(f"⚠️ Missing required data columns for {symbol}", style="bold yellow"))
            return None, None, None
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        if 'Date' in stock_data.columns:
            dates = pd.to_datetime(stock_data['Date'])
        else:
            dates = pd.to_datetime(stock_data.index)
        
        prices = stock_data['Close']
        
        ax.plot(dates, prices, color="#0b273b", linewidth=2.5, alpha=0.9)
        ax.fill_between(dates, prices, alpha=0.3, color='#1f77b4')
        
        ax.set_facecolor("#dacb9f")
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        ax.set_title(f'{company_name} Stock Price - Last {days} Days', 
                    fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xlabel('Date', fontsize=12, color='#34495e')
        ax.set_ylabel('Price ($)', fontsize=12, color='#34495e')
        
        current_price = prices.iloc[-1]
        latest_date = dates.iloc[-1]
        ax.annotate(f'${current_price:.2f}', 
                   xy=(latest_date, current_price),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.8),
                   color='white', fontweight='bold')
        
        plt.tight_layout()
        
        chart_dir = './reports/charts'
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, f'{symbol}_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        console.print(Panel(f"✅ Chart saved: {chart_path}", style="bold green"))
        return chart_path, current_price, symbol
        
    except Exception as e:
        console.print(Panel(f"⚠️ Chart creation failed: {str(e)}", style="bold yellow"))
        import traceback
        console.print(f"Chart error traceback: {traceback.format_exc()}")
        return None, None, None

def add_cell_background_color(cell, color_hex):
    """Helper function to add background color to table cells - FIXED"""
    try:
        # Create shading element properly
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), color_hex)
        tcPr.append(shd)
        return True
    except Exception as e:
        console.print(f"Warning: Could not set cell background color: {e}")
        return False

def create_enhanced_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics):
    """Create a professional DOCX report with enhanced styling - FIXED VERSION"""
    if not DOCX_AVAILABLE:
        console.print(Panel("❌ Cannot create DOCX: python-docx not available", style="bold red"))
        return None
    
    try:
        console.print("📄 Initializing new Word document...")
        doc = Document()
        
        # Set document margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1.25)
            section.right_margin = Inches(1.25)
        
        # Document Header with company name
        title_para = doc.add_paragraph()
        title_run = title_para.add_run('Financial Analysis Report')
        title_run.font.name = 'Arial'  # Changed to more universal font
        title_run.font.size = Pt(24)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(31, 78, 121)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Company name in larger font
        company_para = doc.add_paragraph()
        company_run = company_para.add_run(str(company_name))  # Ensure it's a string
        company_run.font.name = 'Arial'
        company_run.font.size = Pt(20)
        company_run.font.bold = True
        company_run.font.color.rgb = RGBColor(231, 76, 60)
        company_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Date with better formatting
        date_para = doc.add_paragraph()
        date_run = date_para.add_run(f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
        date_run.font.name = 'Arial'
        date_run.font.size = Pt(10)
        date_run.font.italic = True
        date_run.font.color.rgb = RGBColor(127, 140, 141)
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph()  # Space
        
        console.print("📈 Attempting to create and insert chart...")
        chart_path, current_price, symbol = create_price_chart(company_name)
        if chart_path and os.path.exists(chart_path):
            try:
                chart_para = doc.add_paragraph()
                chart_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                chart_run = chart_para.add_run()
                chart_run.add_picture(chart_path, width=Inches(6))
                doc.add_paragraph()  # Space after chart
                console.print("✅ Chart successfully added to document")
            except Exception as e:
                console.print(f"⚠️ Could not add chart to document: {e}")
        
        # Executive Summary
        exec_header = doc.add_paragraph()
        exec_run = exec_header.add_run('📊 Executive Summary')
        exec_run.font.name = 'Arial'
        exec_run.font.size = Pt(16)
        exec_run.font.bold = True
        exec_run.font.color.rgb = RGBColor(52, 73, 94)
        
        exec_body = doc.add_paragraph()
        exec_text = exec_body.add_run(
            'This comprehensive financial analysis synthesizes insights from SEC filings, '
            'real-time market data, and current news sentiment to provide actionable '
            'investment intelligence.'
        )
        exec_text.font.name = 'Arial'
        exec_text.font.size = Pt(11)
        
        # Enhanced Key Metrics Table - FIXED
        if key_metrics and isinstance(key_metrics, dict) and any(key_metrics.values()):
            console.print("📊 Adding key metrics table...")
            
            metrics_header = doc.add_paragraph()
            metrics_run = metrics_header.add_run('💰 Key Financial Metrics')
            metrics_run.font.name = 'Arial'
            metrics_run.font.size = Pt(16)
            metrics_run.font.bold = True
            metrics_run.font.color.rgb = RGBColor(52, 73, 94)
            
            # Create simple table without fancy styling to avoid errors
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'  # Use basic built-in style
            
            # Style header row
            header_cells = table.rows[0].cells
            header_cells[0].text = 'Metric'
            header_cells[1].text = 'Value'
            
            # Make header bold
            for cell in header_cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = 'Arial'
                        run.font.bold = True
                
                # Try to add background color, but continue if it fails
                add_cell_background_color(cell, '1F4E79')
            
            # Add formatted metrics
            metric_labels = {
                'current_price': 'Current Price',
                'market_cap': 'Market Cap',
                'pe_ratio': 'P/E Ratio',
                'eps': 'EPS',
                'week_52_high': '52-Week High',
                'week_52_low': '52-Week Low'
            }
            
            added_rows = 0
            for key, value in key_metrics.items():
                if value is not None and value != 'N/A' and str(value).strip():
                    try:
                        row = table.add_row().cells
                        row[0].text = metric_labels.get(key, key.replace('_', ' ').title())
                        
                        # Format value based on type
                        if key == 'market_cap' and isinstance(value, (int, float)) and value > 0:
                            if value >= 1e12:
                                formatted_value = f"${value/1e12:.2f}T"
                            elif value >= 1e9:
                                formatted_value = f"${value/1e9:.2f}B"
                            else:
                                formatted_value = f"${value/1e6:.2f}M"
                        elif key in ['current_price', 'week_52_high', 'week_52_low'] and isinstance(value, (int, float)):
                            formatted_value = f"${value:.2f}"
                        else:
                            formatted_value = str(value)
                        
                        row[1].text = formatted_value
                        added_rows += 1
                        
                        # Style data cells
                        for cell in row:
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.font.name = 'Arial'
                                    run.font.size = Pt(10)
                    except Exception as e:
                        console.print(f"⚠️ Could not add metric {key}: {e}")
                        continue
            
            if added_rows > 0:
                doc.add_paragraph()
                console.print(f"✅ Added {added_rows} metrics to table")
        
        # Risk Analysis
        risk_header = doc.add_paragraph()
        risk_run = risk_header.add_run('⚠️ Key Business Risks')
        risk_run.font.name = 'Arial'
        risk_run.font.size = Pt(16)
        risk_run.font.bold = True
        risk_run.font.color.rgb = RGBColor(52, 73, 94)
        
        risk_body = doc.add_paragraph()
        risk_content = str(risk_summary) if risk_summary and str(risk_summary).strip() and str(risk_summary) != "None" else "Risk analysis data unavailable at this time. Please check the latest SEC filings for detailed risk factors."
        risk_text = risk_body.add_run(risk_content)
        risk_text.font.name = 'Arial'
        risk_text.font.size = Pt(11)
        
        # Market Performance
        perf_header = doc.add_paragraph()
        perf_run = perf_header.add_run('📈 Market Performance Summary')
        perf_run.font.name = 'Arial'
        perf_run.font.size = Pt(16)
        perf_run.font.bold = True
        perf_run.font.color.rgb = RGBColor(52, 73, 94)
        
        perf_body = doc.add_paragraph()
        clean_performance_text = str(stock_performance_text).replace('**', '') if stock_performance_text else "Performance data unavailable."
        perf_text = perf_body.add_run(clean_performance_text)
        perf_text.font.name = 'Arial'
        perf_text.font.size = Pt(11)
        
        # News Analysis - FIXED
        news_header = doc.add_paragraph()
        news_run = news_header.add_run('📰 Recent News & Market Catalysts')
        news_run.font.name = 'Arial'
        news_run.font.size = Pt(16)
        news_run.font.bold = True
        news_run.font.color.rgb = RGBColor(52, 73, 94)
        
        if news_articles and isinstance(news_articles, list) and len(news_articles) > 0:
            news_intro = doc.add_paragraph()
            intro_text = news_intro.add_run('Recent market developments include:')
            intro_text.font.name = 'Arial'
            intro_text.font.size = Pt(11)
            
            for i, article in enumerate(news_articles[:5], 1):
                if isinstance(article, dict) and article.get('title'):
                    news_item = doc.add_paragraph()
                    # Add bullet manually
                    bullet_run = news_item.add_run('• ')
                    bullet_run.font.name = 'Arial'
                    bullet_run.font.size = Pt(10)
                    
                    title = str(article.get('title', 'N/A'))
                    item_text = news_item.add_run(title)
                    item_text.font.name = 'Arial'
                    item_text.font.size = Pt(10)
        else:
            news_body = doc.add_paragraph()
            news_text = news_body.add_run('Recent market developments and analyst coverage continue to evolve. Monitor financial news for latest updates.')
            news_text.font.name = 'Arial'
            news_text.font.size = Pt(11)
        
        # Professional Disclaimer
        doc.add_paragraph()
        disclaimer_header = doc.add_paragraph()
        disclaimer_run = disclaimer_header.add_run('⚖️ Important Disclaimer')
        disclaimer_run.font.name = 'Arial'
        disclaimer_run.font.size = Pt(14)
        disclaimer_run.font.bold = True
        disclaimer_run.font.color.rgb = RGBColor(192, 57, 43)
        
        disclaimer_body = doc.add_paragraph()
        disclaimer_text = disclaimer_body.add_run(
            'This report is generated for informational and educational purposes only. It should not be '
            'considered as personalized investment advice, a recommendation to buy or sell securities, '
            'or a guarantee of future performance. All investments carry risk of loss. Past performance '
            'does not guarantee future results. Please consult with a qualified financial advisor '
            'before making any investment decisions.'
        )
        disclaimer_text.font.name = 'Arial'
        disclaimer_text.font.size = Pt(9)
        disclaimer_text.font.italic = True
        disclaimer_text.font.color.rgb = RGBColor(127, 140, 141)
        
        console.print("✅ Document structure created successfully")
        return doc
        
    except Exception as e:
        console.print(Panel(f"❌ Error creating document: {str(e)}", style="bold red"))
        import traceback
        console.print(f"Document creation traceback: {traceback.format_exc()}")
        return None

def curate_report_node(state: FinanceAgentState) -> dict:
    """
    Synthesizes information from multiple sources to generate a comprehensive
    financial analyst report and returns it as a Markdown string.
    """
    
    console.print(Panel("🔄 NODE: Generating Professional Analyst Report", style="bold green"))

    # --- 1. RETRIEVE DATA FROM CORRECT CHANNELS WITH BETTER ERROR HANDLING ---
    company_name = state.get("company_name", "Unknown Company")
    
    # Safely retrieve data with fallbacks
    risk_summary = state.get("tool_result")
    if not risk_summary or risk_summary == "None" or str(risk_summary).strip() == "":
        risk_summary = "Risk analysis data not available from SEC filings."
    
    key_metrics = state.get("structured_data", {})
    if not isinstance(key_metrics, dict):
        key_metrics = {}
    
    news_articles = state.get("news_results", [])
    if not isinstance(news_articles, list):
        news_articles = []
    
    console.print(f"📊 Company: {company_name}")
    console.print(f"📈 Key Metrics Available: {len(key_metrics) if key_metrics else 0}")
    console.print(f"📰 News Articles: {len(news_articles)}")
    console.print(f"⚠️ Risk Summary Length: {len(str(risk_summary)) if risk_summary else 0}")
    
    # Format stock performance summary for the report body
    if key_metrics and isinstance(key_metrics, dict) and any(key_metrics.values()):
        try:
            current_price = key_metrics.get('current_price', 'N/A')
            market_cap = key_metrics.get('market_cap', 'N/A')
            pe_ratio = key_metrics.get('pe_ratio', 'N/A')
            
            # Format market cap properly
            if isinstance(market_cap, (int, float)) and market_cap > 0:
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = str(market_cap)
            
            console.print(f"Stock performance metrics")

            stock_performance_text = (
                f"Current Price: **${current_price}** | "
                f"Market Cap: **{market_cap_str}** | "
                f"P/E Ratio: **{pe_ratio}**"
            )
        except Exception as e:
            console.print(f"⚠️ Error formatting metrics: {e}")
            stock_performance_text = "Stock performance data is available but formatting failed."
    else:
        stock_performance_text = "Stock performance data is unavailable."

    if company_name == "Unknown Company":
        error_msg = "Failed to generate a report. Could not determine the company from your query."
        console.print(Panel(error_msg, style="bold red"))
        return {
            "final_answer": error_msg,
            "messages": [AIMessage(content=error_msg)]
            }

    if news_articles and isinstance(news_articles, list) and len(news_articles) > 0:
        try:
            news_items = []
            for article in news_articles[:5]:  # Limit to 5 articles
                if isinstance(article, dict) and article.get('title'):
                    title = str(article.get('title', 'N/A'))
                    if title and title != 'N/A':
                        news_items.append(f"- **{title}**")
            
            if news_items:
                news_summary_text = "\n".join(news_items)
            else:
                news_summary_text = f"- **{company_name}** market developments ongoing\n- Analyst coverage and market sentiment analysis continues"
        except Exception as e:
            console.print(f"⚠️ Error formatting news: {e}")
            news_summary_text = f"- **{company_name}** market developments ongoing\n- Analyst coverage and market sentiment analysis continues"
    else:
        news_summary_text = f"- **{company_name}** market developments ongoing\n- Analyst coverage and market sentiment analysis continues"

    # ---REPORT CONSTRUCT---
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
    
    # Create and save enhanced DOCX - COMPLETELY REWRITTEN FOR RELIABILITY
    saved_successfully = False
    docx_path = None
    
    if DOCX_AVAILABLE:
        try:
            # Ensure reports directory exists
            reports_dir = './reports'
            os.makedirs(reports_dir, exist_ok=True)
            abs_reports_dir = os.path.abspath(reports_dir)
            console.print(f"📁 Reports directory: {abs_reports_dir}")
            
            # Check directory permissions
            if not os.access(reports_dir, os.W_OK):
                console.print(Panel("❌ No write permission to reports directory", style="bold red"))
            else:
                console.print("✅ Write permission confirmed")
            
            # Generate enhanced DOCX
            console.print("📄 Creating DOCX document...")
            doc = create_enhanced_docx_report(company_name, risk_summary, stock_performance_text, news_articles, key_metrics)
            
            if doc:  # Only save if doc was created successfully
                # Generate safe filename
                safe_company_name = "".join(c for c in str(company_name) if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_company_name = safe_company_name.replace(' ', '_')[:50]  # Limit length
                if not safe_company_name:
                    safe_company_name = "Company_Report"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_company_name}_Report_{timestamp}.docx"
                docx_path = os.path.join(reports_dir, filename)
                
                console.print(f"💾 Saving document to: {os.path.abspath(docx_path)}")
                
                # Save the document
                doc.save(docx_path)
                
                # Verify the file was saved
                if os.path.exists(docx_path):
                    file_size = os.path.getsize(docx_path)
                    console.print(Panel(f"✅ DOCX saved successfully!\nPath: {os.path.abspath(docx_path)}\nSize: {file_size:,} bytes", style="bold green"))
                    saved_successfully = True
                else:
                    console.print(Panel("❌ File was not saved (file does not exist after save)", style="bold red"))
            else:
                console.print(Panel("❌ Document creation failed - doc object is None", style="bold red"))
            
        except Exception as e:
            console.print(Panel(f"❌ DOCX save failed: {str(e)}", style="bold red"))
            import traceback
            console.print(f"Full DOCX error traceback: {traceback.format_exc()}")
    else:
        console.print(Panel("📄 DOCX export skipped (python-docx not available)", style="bold yellow"))

    console.print("---NODE: Report Generation Complete---")
    
    try:
        from IPython.display import Markdown, display
        display(Markdown(report_content_md))
    except ImportError:
        pass  # Not in Jupyter environment
    
    print(RichMarkdown(f"DEBUG : {state.get('final_answer')}"))

    return {
        "report_data": {
            "company_name": company_name,
            "risks": risk_summary,
            "performance_metrics": stock_performance_text,
            "news_count": len(news_articles) if isinstance(news_articles, list) else 0
        },
        "final_answer": report_content_md, 
        "messages":[
            AIMessage(content=f"Here's your final report for {company_name}. It includes key risks, performance metrics, and related news. ")
        ],
    }