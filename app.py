import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

#load imdb word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Load the pre-trained model
model = load_model('simple_rnn_model_imdb.h5')

#decode the review text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

#encode the review
def process_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    encoded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return encoded_review

#sentiment prediction of moview
def predict_sentiment(review):
    processed_review = process_review(review)
    prediction = model.predict(processed_review)[0][0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment, prediction


#streamlit app UI
st.title("IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

user_input = st.text_input("Enter a movie review:")

if st.button("Analyze"):
    if user_input:
        sentiment, prediction = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction}")
    else:
        st.warning("Please enter a movie review.")

