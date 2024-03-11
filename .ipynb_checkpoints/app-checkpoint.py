# Import some required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from collections import Counter
import contractions
from nrclex import NRCLex
from textblob import TextBlob
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# Spacy
import spacy
# Load the pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')

import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import joblib

# Load necessary objects
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
numerical_columns = ['count_positive_words', 'count_negative_words', 'contain_no', 'contain_not',
                     'contain_exclamation', 'log_review_length', 'emotion_label', 'sentiment_score']

# Load the SVM classifier
svm_classifier = joblib.load('best_model.pkl')  # Load your trained SVM model

def text_preprocessing(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Replace repetitions of punctuation with a single punctuation mark
    text = re.sub(r'(\W)\1+', r'\1', text)

    # Remove punctuation(except !) and numbers
    text = re.sub(r'[^a-zA-Z!]+', ' ', text)

    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Tokenize text using SpaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Remove stop words except 'no' and 'not' - SpaCy has its own stop words list
    tokens = [token for token in tokens if not nlp.vocab[token].is_stop or token in {'no', 'not'}]

    # Lemmatization - SpaCy performs lemmatization automatically
    tokens = [token.lemma_ for token in doc]

    # Return Join tokens back into text
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def count_positive_words(cleaned_review_text):
    positive_words = set()
    with open('Datasets/positive-words.txt', 'r', encoding='latin-1') as f:
        positive_words = set(f.read().splitlines())
    words = cleaned_review_text.split()
    count = sum(1 for word in words if word in positive_words)
    return count

def count_negative_words(cleaned_review_text):
    negative_words = set()
    with open('Datasets/negative-words.txt', 'r', encoding='latin-1') as f:
        negative_words = set(f.read().splitlines())
    words = cleaned_review_text.split()
    count = sum(1 for word in words if word in negative_words)
    return count

def contain_no(cleaned_review_text):
    tokens = cleaned_review_text.split()
    contain_no = int('no' in tokens)
    return contain_no

def contain_not(cleaned_review_text):
    tokens = cleaned_review_text.split()
    contain_not = int('not' in tokens)
    return contain_not

def contain_exclamation(cleaned_review_text):
    tokens = cleaned_review_text.split()
    contain_exclamation = int('!' in tokens)
    return contain_exclamation

def log_review_length(review_text):
    review_length = len(review_text)
    if review_length == 0:
        return 0  # Return 0 if the review length is 0
    log_length = math.log(review_length)
    return log_length

def get_emotion_label(phrase):
    # Tokenize the phrase
    tokens = phrase.split()

    # Define emotion word list
    emotion_words = []

    for i in range(len(tokens)):
      # Call NRCLex constructor
      emotion = NRCLex(tokens[i])
      # Get affect_dict
      affect_dict = emotion.affect_dict
      # Extract emotion words from affect_dict values
      for sublist in affect_dict.values():
        emotion_words.extend(sublist)

    # Define positive and negative emotion groups
    positive_emotions = ['anticipation', 'trust', 'positive', 'joy', 'surprise']
    negative_emotions = ['fear', 'anger', 'negative', 'sadness', 'disgust']

    # Count occurrences of positive and negative words
    positive_count = sum(emotion_words.count(emotion) for emotion in positive_emotions)
    negative_count = sum(emotion_words.count(emotion) for emotion in negative_emotions)

    # Determine sentiment label based on counts
    if positive_count > negative_count:
        return 1
    elif positive_count < negative_count:
        return 0
    else:
        return 2

def calculate_sentiment_score(phrase):
    # Create a TextBlob object
    blob = TextBlob(phrase)

    # Get the sentiment score
    sentiment_score = blob.sentiment.polarity

    return sentiment_score

# Define functions for feature extraction 
def extract_features(cleaned_text):
    count_pos_words = count_positive_words(cleaned_text)
    count_neg_words = count_negative_words(cleaned_text)
    contain_no_val = contain_no(cleaned_text)
    contain_not_val = contain_not(cleaned_text)
    contain_exclamation_val = contain_exclamation(cleaned_text)
    log_length = log_review_length(cleaned_text)
    emotion_label = get_emotion_label(cleaned_text)
    sentiment_score = calculate_sentiment_score(cleaned_text)
    return [count_pos_words, count_neg_words, contain_no_val, contain_not_val, contain_exclamation_val, log_length, emotion_label, sentiment_score]

# Define functions for sentiment prediction
def predict_sentiment(cleaned_text):
    features = extract_features(cleaned_text)
    df = pd.DataFrame([features], columns=numerical_columns)
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    tfidf_text = tfidf_vectorizer.transform([cleaned_text])
    X = hstack([tfidf_text, csr_matrix(df[numerical_columns].values)])
    prediction = svm_classifier.predict(X)
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# Streamlit web app
def main():
    st.title("Sentiment Analysis")
    input_text = st.text_input("Enter your text:")
    if st.button("Analyze"):
        if input_text:
            cleaned_text = text_preprocessing(input_text)
            predicted_sentiment = predict_sentiment(cleaned_text)
            st.write("Predicted sentiment:", predicted_sentiment)

if __name__ == "__main__":
    main()
