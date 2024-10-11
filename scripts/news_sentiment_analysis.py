import sys
import os

# add project root to sys.path manually
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"sys.path in news_sentiment_analysis: {sys.path}")

import joblib
import pandas as pd
from .news_fetcher import get_news_articles

# load traned model and vectorizer
model = joblib.load('../models/sentiment_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# analyze sentiment of news articles
def analyze_sentiment_for_news(articles_df):
    # preprocess article content
    articles_df['cleaned_content'] = articles_df['content'].fillna('').str.lower()

    # vectorize content 
    vectorized_content = vectorizer.transform(articles_df['cleaned_content'])

    # predict sentiment using loaded model
    articles_df['sentiment'] = model.predict(vectorized_content)
    
    # map sentiment (0 = negative, 4 = positive)
    articles_df['sentiment_label'] = articles_df['sentiment'].map({0: 'Negative', 4: 'Positive'})
    
    print(articles_df[['cleaned_content', 'sentiment_label']].head(30))

    return articles_df[['title', 'publishedAt', 'sentiment_label']]

if __name__ == '__main__':
    # fetch articles using news_fetcher.py function
    query = 'artificial intelligence'  # update query for each use case
    # NewsAPI restricts free tier access to 1 month prior to runtime
    # change date range as required (in case of 426 error)
    from_date = '2024-10-01'
    to_date = '2024-10-04'
    
    # fetch news articles
    articles_df = get_news_articles(query, from_date, to_date)
    
    # analyze sentiment if articles were fetched successfully
    if articles_df is not None:
        sentiment_df = analyze_sentiment_for_news(articles_df)
        print(sentiment_df.head())