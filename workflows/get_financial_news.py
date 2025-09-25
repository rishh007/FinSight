# get_financial_news.py (CORRECTED)

from newsapi.newsapi_client import NewsApiClient
from bs4 import BeautifulSoup
from state import FinanceAgentState
import requests 
import os
import json # Ensure json is imported for consistency if you use it later

def get_financial_news_node(state: FinanceAgentState) -> dict:
    """
    Fetches recent financial news articles for a company name found in the state.
    It updates the state with the list of news articles and a confirmation message.
    """
    # --- INSERT THIS LOGIC AT THE END OF EVERY NODE ---
    if state.get("final_answer"):
        print("\n" + state["final_answer"])
        print("-" * 30)
    # --------------------------------------------------
    print("\n---NODE: Getting Financial News---")
    
    # Get the company name from the state (should be populated by 'Extract Entities')
    company_name = state.get("company_name")

    # CRITICAL FIX: If company_name is missing, do NOT call input().
    # Return a message that loops back to the user.
    if not company_name:
        error_message = "Error: Cannot fetch news. The company name or ticker was not found in your previous query."
        print(error_message)
        return {"final_answer": error_message}
    
    # Ensure API key is available
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        return {"final_answer": "Error: NEWS_API_KEY is not set in environment variables."}

    newsapi = NewsApiClient(api_key=news_api_key)
    
    try:
        # 1. Fetch top headlines
        top_headlines = newsapi.get_everything(
            q=company_name,
            language='en',
            sort_by='publishedAt',
            page_size=2 
        )
        
        articles_summary = []
        for article in top_headlines.get('articles', []):
            try:
                # 2. Simple scrape to get a content snippet
                response = requests.get(article['url'], timeout=5)
                soup = BeautifulSoup(response.content, 'html.parser')
                first_paragraph = soup.find('p').get_text() if soup.find('p') else "No content preview available."
                
                articles_summary.append({
                    "title": article['title'],
                    "url": article['url'],
                    "content_snippet": first_paragraph[:250] + "..." # Truncate for brevity
                })
            except Exception as e:
                print(f"Skipping article {article['url']} due to error: {e}")
                continue
                
        # 3. Update the state with the results
        success_message = f"Successfully fetched {len(articles_summary)} news articles for {company_name}."
        print(success_message)
        return {
            "news_results": articles_summary,
            "final_answer": success_message
        }
        
    except Exception as e:
        # This handles API errors (429 rate limit, 400 bad request, etc.)
        error_message = f"Failed to fetch news (API Error). Ensure your API key is correct. Error: {e}"
        print(error_message)
        return {"final_answer": error_message}