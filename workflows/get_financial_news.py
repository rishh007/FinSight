import requests
from bs4 import BeautifulSoup
from state import FinanceAgentState
import os
from datetime import datetime, timedelta
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_core.messages import AIMessage, HumanMessage 

console = Console()
from dotenv import load_dotenv
load_dotenv()
def get_financial_news_node(state: FinanceAgentState) -> dict:
    """
    Fetches recent financial news articles for a company name found in the state.
    It updates the state with the list of news articles and a confirmation message.
    """
    
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
    search_table.add_row("Language", "English")
    search_table.add_row("Sort By", "Published Date")
    search_table.add_row("Articles Limit", "2")
    
    console.print(search_table)
    
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        error_message = "Error: NEWS_API_KEY is not set in environment variables."
        console.print(Panel(f"❌ {error_message}", style="bold red"))
        return {"final_answer": error_message}
    
    console.print("🔐 News API key found - preparing request...")
    
    try:
        headers = {'Authorization': f'Bearer {news_api_key}'}
        params = {
            'q': company_name,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 2
        }
        
        with console.status(f"[bold green]Searching for news articles about {company_name}..."):
            response = requests.get(
                'https://newsapi.org/v2/everything',
                headers=headers,
                params=params
            )
            top_headlines = response.json()
        
        articles_found = top_headlines.get('articles', [])
        console.print(f"📈 Found {len(articles_found)} articles from News API")
        
        if not articles_found:
            message = f"No recent news articles found for {company_name}."
            console.print(Panel(f"⚠️ {message}", style="bold yellow"))
            return {"final_answer": message}
        
        articles_summary = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing articles...", total=len(articles_found))
            
            for i, article in enumerate(articles_found, 1):
                progress.update(task, description=f"Processing article {i}/{len(articles_found)}")
                
                try:
                    response = requests.get(article['url'], timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    content_snippet = None
                    
                    # Method 1: First paragraph
                    first_p = soup.find('p')
                    if first_p and first_p.get_text().strip():
                        content_snippet = first_p.get_text().strip()
                    
                    # Method 2: Meta description as fallback
                    if not content_snippet:
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc and meta_desc.get('content'):
                            content_snippet = meta_desc.get('content')
                    
                    # Method 3: Article description from API as final fallback
                    if not content_snippet:
                        content_snippet = article.get('description', 'No content preview available.')
                    
                    articles_summary.append({
                        "title": article['title'],
                        "url": article['url'],
                        "published_at": article.get('publishedAt', 'Unknown'),
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "content_snippet": content_snippet[:250] + "..." if len(content_snippet) > 250 else content_snippet
                    })
                    
                    progress.advance(task)
                    
                except requests.RequestException as e:
                    console.print(f"⚠️ Could not scrape content for article: {e}")
                    articles_summary.append({
                        "title": article['title'],
                        "url": article['url'],
                        "published_at": article.get('publishedAt', 'Unknown'),
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "content_snippet": article.get('description', 'Content not available.')[:250] + "..."
                    })
                    progress.advance(task)
                    continue
                
                except Exception as e:
                    console.print(f"⚠️ Skipping article due to error: {e}")
                    progress.advance(task)
                    continue
        
        if articles_summary:
            results_table = Table(title="📰 News Articles Found", show_header=True, header_style="bold green")
            results_table.add_column("Title", style="white", max_width=40)
            results_table.add_column("Source", style="cyan", max_width=15)
            results_table.add_column("Published", style="yellow", max_width=12)
            results_table.add_column("Preview", style="bright_black", max_width=50)
            
            for article in articles_summary:
                pub_date = article['published_at']
                if pub_date and pub_date != 'Unknown':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%m/%d/%Y')
                    except:
                        formatted_date = pub_date[:10]  
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
                    f"[dim]Source: {first_article['source']} | Published: {formatted_date}[/dim]\n\n"
                    f"{first_article['content_snippet']}",
                    title="📄 Featured Article Preview",
                    title_align="left",
                    border_style="blue",
                    padding=(1, 2)
                ))
        
        stats_table = Table(title="News Extraction Summary", show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Company Searched", company_name)
        stats_table.add_row("Articles Found", str(len(articles_found)))
        stats_table.add_row("Articles Processed", str(len(articles_summary)))
        stats_table.add_row("Success Rate", f"{(len(articles_summary)/len(articles_found)*100):.0f}%" if articles_found else "0%")
        
        console.print(stats_table)
        
        success_message = f"Successfully fetched {len(articles_summary)} news articles for {company_name}."
        console.print(Panel(f"✅ {success_message}", style="bold green"))
        
        return {
            "news_results": articles_summary,
            "final_answer": success_message,
            "messages": [
                AIMessage(content=f"I found {len(articles_summary)} recent news articles about {company_name}. Here’s a brief summary of the highlights.")
                ]
        }
        
    except Exception as e:
        error_message = f"Failed to fetch news (API Error). Ensure your API key is correct. Error: {e}"
        console.print(Panel(f"❌ {error_message}", style="bold red"))
        
        if "429" in str(e):
            console.print(Panel(
                "Rate limit exceeded. Consider:\n"
                "• Waiting before retrying\n"
                "• Upgrading your News API plan\n"
                "• Reducing request frequency",
                title="💡 Rate Limit Help",
                border_style="yellow"
            ))
        elif "401" in str(e):
            console.print(Panel(
                "Authentication failed. Check:\n"
                "• NEWS_API_KEY is correctly set\n"
                "• API key is valid and active\n"
                "• No extra spaces in the key",
                title="🔑 Authentication Help",
                border_style="yellow"
            ))
        
        return {"final_answer": error_message, 
                "messages": [AIMessage(content=error_message)]}