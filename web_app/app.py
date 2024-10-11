from flask import Flask, render_template, request, jsonify
import sys
import os

# add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("sys.path:", sys.path)
print("Current working directory:", os.getcwd())

from scripts.news_fetcher import get_news_articles
from scripts.news_sentiment_analysis import analyze_sentiment_for_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentiment_data', methods=['POST'])
def get_sentiment_data():
    # get user input from the JSON request body
    user_input = request.get_json()
    query = user_input['topic']
    from_date = user_input['from_date']
    to_date = user_input['to_date']
    
    # fetch & analyze news articles
    articles_df = get_news_articles(query, from_date, to_date)

    if articles_df is None:
        return jsonify({'error': 'No articles found.'}), 404
    
    sentiment_df = analyze_sentiment_for_news(articles_df)
    data = sentiment_df.to_dict(orient='records')

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
