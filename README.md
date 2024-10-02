# SentimentAI

## Overview

This project performs sentiment analysis on tweets using the **Sentiment140** dataset. The goal is to classify tweets as **positive** or **negative** based on their content. The project includes data preprocessing, machine learning model training, and optional deployment via a web application.

## Dataset

- The dataset used in this project is **Sentiment140**, which consists of 1.6 million tweets labeled for sentiment.
- The data is stored in the `/data/` directory. The raw dataset is located in `data/raw/sentiment140.csv`.
- Preprocessing scripts clean the dataset and output the results to `data/processed/`.

## Features

- **Text Preprocessing**: Tokenization, removal of URLs, special characters, and stopwords.
- **Sentiment Analysis**: A machine learning model is trained to classify tweets as positive or negative.
- **Deployment**: (Optional) A simple Flask web application is included for deploying the model as a web service.

## Setup and Installation

### Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

###  Data Preprocessing

To preprocess the data, run the following script:

```
python scripts/data_preprocessing.py
```
This will clean the raw data and store it in data/processed/.

### Model Training
To train the sentiment analysis model, run:
```
python scripts/train_model.py
```
This script will train a machine learning model on the preprocessed data and save the trained model to models/sentiment_model.pkl.

### Predictions
To use the trained model to predict the sentiment of new text:
```
python scripts/predict.py
```
You can modify the predict.py script to input new tweets or texts.

## Running the Flask App
If you want to deploy the model via a web application, you can run the Flask app:
```
python web_app/app.py
```
Access the app in your browser at http://127.0.0.1:5000/.

## Tests
Run the unit tests with the following command:
```
python -m unittest discover tests/
```
## Future Improvements
- Improve the sentiment analysis model by using more advanced techniques like fine-tuning BERT.

- Add support for neutral sentiment (currently a binary classification model).

- Deploy the model on cloud services (AWS Lambda, Heroku) for wider accessibility.

## License
This project is open-source and available under the MIT License.

