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

# preprocess dataset in chunks to accommodate weaker systems
def preprocess_data():
    chunk_size = 100000 # change this to accomomodate your system
    raw_data_path = '../data/raw/sentiment140.csv'
    processed_data_path = '../data/processed/cleaned_data.csv'
    
    # create empty CSV file to store processed data
    with open(processed_data_path, 'w') as f:
        pass

    for chunk in pd.read_csv(raw_data_path, encoding='ISO-8859-1', header=None, chunksize=chunk_size):
        # assign column names to chunk
        chunk.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        # clean text in chunk
        chunk['cleaned_text'] = chunk['text'].apply(clean_tweet)
        
        # save processed chunk to processed data file
        chunk[['cleaned_text', 'target']].to_csv(processed_data_path, mode='a', header=False, index=False)
        
        print(f'Processed chunk of size {chunk_size} rows.')

if __name__ == '__main__':
    preprocess_data()