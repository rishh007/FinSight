import requests
from state import FinanceAgentState
import os
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

console = Console()
load_dotenv()

def get_financial_news_node(state: FinanceAgentState) -> dict:
    console.print(Panel("📰 NODE: Getting Financial News", style="bold cyan"))
    
    company_name = state.get("company_name")
    if not company_name:
        error_message = "Error: Cannot fetch news. The company name or ticker was not found in your previous query."
        console.print(Panel(f"❌ {error_message}", style="bold red"))
        return {"final_answer": error_message}
    
    search_table = Table(title="News Search Parameters", show_header=True, header_style="bold magenta")
    search_table.add_column("Parameter", style="cyan")
    search_table.add_column("Value", style="green")
    search_table.add_row("Company Name", company_name)
    search_table.add_row("Search Type", "News")
    search_table.add_row("Results Limit", "5")
    console.print(search_table)
    
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        error_message = "Error: SERPER_API_KEY is not set in environment variables."
        console.print(Panel(f"❌ {error_message}", style="bold red"))
        return {"final_answer": error_message}
    
    console.print("🔐 Serper API key found - preparing request...")
    
    try:
        with console.status(f"[bold green]Searching for news articles about {company_name}..."):
            response = requests.post(
                'https://google.serper.dev/news',
                headers={'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'},
                json={'q': f'{company_name} stock news', 'num': 5}
            )
            response.raise_for_status()
            data = response.json()
        
        news_results = data.get('news', [])
        console.print(f"📈 Found {len(news_results)} articles from Serper API")
        
        if not news_results:
            message = f"No recent news articles found for {company_name}."
            console.print(Panel(f"⚠️ {message}", style="bold yellow"))
            return {"final_answer": message}
        
        articles_summary = []
        for article in news_results:
            articles_summary.append({
                "title": article.get('title', 'No title'),
                "url": article.get('link', ''),
                "published_at": article.get('date', 'Unknown'),
                "source": article.get('source', 'Unknown'),
                "content_snippet": article.get('snippet', 'No preview available.')
            })
        
        results_table = Table(title="📰 News Articles Found", show_header=True, header_style="bold green")
        results_table.add_column("Title", style="white", max_width=40)
        results_table.add_column("Source", style="cyan", max_width=15)
        results_table.add_column("Published", style="yellow", max_width=12)
        results_table.add_column("Preview", style="bright_black", max_width=50)
        
        for article in articles_summary:
            pub_date = article['published_at']
            if pub_date and pub_date != 'Unknown':
                try:
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    formatted_date = dt.strftime('%m/%d/%Y')
                except:
                    formatted_date = pub_date[:10] if len(pub_date) >= 10 else pub_date
            else:
                formatted_date = 'Unknown'
            
            results_table.add_row(
                article['title'][:40] + "..." if len(article['title']) > 40 else article['title'],
                article['source'][:15] + "..." if len(article['source']) > 15 else article['source'],
                formatted_date,
                article['content_snippet'][:50] + "..." if len(article['content_snippet']) > 50 else article['content_snippet']
            )
        
        console.print(results_table)
        
        if articles_summary:
            first_article = articles_summary[0]
            console.print(Panel(
                f"[bold]{first_article['title']}[/bold]\n\n"
                f"[dim]Source: {first_article['source']}[/dim]\n\n"
                f"{first_article['content_snippet']}",
                title="📄 Featured Article Preview",
                title_align="left",
                border_style="blue",
                padding=(1, 2)
            ))
        
        success_message = f"Successfully fetched {len(articles_summary)} news articles for {company_name}."
        console.print(Panel(f"✅ {success_message}", style="bold green"))
        
        return {
            "news_results": articles_summary,
            "final_answer": success_message,
            "messages": [AIMessage(content=f"I found {len(articles_summary)} recent news articles about {company_name}.")]
        }
        
    except Exception as e:
        error_message = f"Failed to fetch news. Error: {e}"
        console.print(Panel(f"❌ {error_message}", style="bold red"))
        return {"final_answer": error_message, "messages": [AIMessage(content=error_message)]}
