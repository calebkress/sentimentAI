import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# load preprocessed data
def load_preprocessed_data():
    processed_data_path = '../data/processed/cleaned_data.csv'
    df = pd.read_csv(processed_data_path)
    return df

# train model
def train_model():
    # load cleaned data
    df = load_preprocessed_data()

    # features (X) and target (y)
    X = df['cleaned_text']
    y = df['target']

    # split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # convert text data to numberical features
    vectorizer = TfidfVectorizer(max_features = 5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # initialize & train logistic regression model
    model = LogisticRegression(max_iter = 200)
    model.fit(X_train_tfidf, y_train)

    # make predictions on test set
    y_pred = model.predict(X_test_tfidf)

    # print model evaluation metrics
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # save trained model & vectorizer for later use
    joblib.dump(model, '../models/sentiment_model.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved to ../models/")

if __name__ == '__main__':
    train_model()