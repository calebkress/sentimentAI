from flask import Flask, render_template, jsonify
from scripts.news_fetcher import get_news_articles
from scripts.news_sentiment_analysis import analyze_sentiment_for_news


app = Flask(__name__)

@app.route('/get_sentiment_data')
def get_sentiment_data():
    # TODO: replace these values with user data
    query = 'artificial intelligence'
    from_date = '2024-09-15'
    to_date = '2024-10-07'
    
    # fetch & analyze news articles
    articles_df = get_news_articles(query, from_date, to_date)

    if articles_df is None:
        return jsonify({'error': 'No articles found.'}), 404
    
    sentiment_df = analyze_sentiment_for_news(articles_df)
    data = sentiment_df.to_dict(orient='records')

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)