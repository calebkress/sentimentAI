import requests
import pandas as pd
from dotenv import load_dotenv
import os

# load .env
load_dotenv()

# get API key from .env
api_key = os.getenv('NEWSAPI_KEY')

# check if API key is loaded correctly
if not api_key:
    raise ValueError("API key not found. Check your .env file.")


# fetch news articles
def get_news_articles(query, from_date, to_date, page_size=100):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'sortBy': 'relevancy',
        'pageSize': page_size,
        'apiKey': api_key
    }

    # print API key to verify it's being used (for debugging)
    print(f"Using API Key: {api_key}")

    response = requests.get(url, params=params)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        # convert articles into DataFrame
        data = [{'title': article['title'], 'description': article['description'], 'content': article['content'], 'publishedAt': article['publishedAt']} for article in articles]
        return pd.DataFrame(data)
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

