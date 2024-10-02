import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import ssl

# bypass SSL verification (fix for SSL cert error)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# download stopwords from NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# define stopwords set
stop_words = set(stopwords.words('english'))

# fn to clean individual tweets
def clean_tweet(tweet):
    # remove urls and special characters
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    # remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    
    # convert to lowercase
    tweet = tweet.lower()
    
    # tokenize tweet into words
    words = word_tokenize(tweet)
    
    # remove stopwords
    cleaned_words = [word for word in words if word not in stop_words]
    
    # rejoin cleaned words into single string
    return ' '.join(cleaned_words)

# load raw dataset
def load_data():
    raw_data_path = '../data/raw/sentiment140.csv'

    # load with no header
    try:
        df = pd.read_csv(raw_data_path, encoding='ISO-8859-1', header=None)
        # Assign column names to the dataset
        df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        print(f"Data loaded successfully with {df.shape[0]} rows.")
        return df
    except FileNotFoundError:
        print(f"File not found: {raw_data_path}. Please check the file path.")
        return None
    # assign column names to dataset
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    return 

# preprocess dataset
def preprocess_data():
    # load raw data
    df = load_data()
    
    # clean tweet text
    df['cleaned_text'] = df['text'].apply(clean_tweet)

    # drop unnecessary columns
    df_processed = df[['cleaned_text', 'target']]
    
    # save processed data to processed folder
    processed_data_path = '../data/processed/cleaned_data.csv'
    df_processed.to_csv(processed_data_path, index=False)
    print(f'Processed data saved to {processed_data_path}')

if __name__ == '__main__':
    preprocess_data()