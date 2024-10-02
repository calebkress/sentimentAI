import joblib

# load trained model and vectorizer
model = joblib.load('../models/sentiment_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# fn to predict sentiment for new text
def predict_sentiment(text):
    # preprocess input text
    processed_text = text.lower()
    # vectorize input text
    vectorized_text = vectorizer.transform([processed_text])
    # make prediction using loaded model
    prediction = model.predict(vectorized_text)

    # map prediction to sentiment label (4 = positive, 0 = negative)
    if prediction == 0:
        return "Negative sentiment"
    elif prediction == 4:
        return "Positive sentiment"
    
if __name__ == '__main__':
    text = input("Enter text to analyze sentiment: ")
    sentiment = predict_sentiment(text)
    print(f"Predicted sentiment: {sentiment}")