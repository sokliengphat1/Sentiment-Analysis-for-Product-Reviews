# Sentiment-Analysis-for-Product-Reviews

This mini project implements binary text classification techniques to perform sentiment analysis on product reviews. Given a corpus of positive and negative reviews, the goal is to build a classifier to predict the sentiment (positive or negative) of a given review.

### Directories:

The project directory contains code and data for sentiment analysis. Below is the structure of directories and files:

#### Datasets/

- **negative-reviews.txt**: Contains a collection of negative reviews.
- **positive-reviews.txt**: Contains a collection of positive reviews.
- **positive-words**: Possibly a file containing a list of positive words used for sentiment analysis.
- **negative-words**: Possibly a file containing a list of negative words used for sentiment analysis.

#### Models/

- **best_model.pkl**: A saved machine learning model (possibly for sentiment analysis) in the pickle format.
- **scaler.pkl**: A saved scaler object (possibly used for scaling features) in the pickle format.
- **tfidf_vectorizer**: A saved TF-IDF vectorizer object (possibly used for text vectorization) in an appropriate format.

### Files:

- **app.py**: This is the main Python file containing the Streamlit application for sentiment analysis.
- **Sentiment_Classification_Analysis.ipynb**: This is a Jupyter Notebook file that contains code for sentiment analysis, including data preprocessing, feature extraction, model training, and evaluation.

## Installation Requirements

To run the code in this project, you will need the following libraries:

- pandas
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- numpy
- spaCy
- nltk
- textblob 
- nrclex
- constractions
- streamlit 
- scipy
- joblib
- scikit-learn

Additionally, if you're using spaCy, you'll need to download the English language model:
- python -m spacy download en_core_web_sm

🚀 **Test the Streamlit app [here](https://sentiment-analysis-for-appuct-reviews-9ysxxd4btrqvzvx5zuv8ya.streamlit.app/)** 🌟



