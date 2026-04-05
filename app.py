import streamlit as st
import pickle
import string
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# UI
st.title("📰 Fake News Detection System")
st.info("Note: This is a beginner-level machine learning model developed for learning purposes. Predictions may not always be accurate.")
input_text = st.text_area("Enter News Text")

if st.button("Predict"):
    processed = preprocess(input_text)
    vector = vectorizer.transform([processed])
    result = model.predict(vector)
    
    st.subheader("Result:")
    st.success(result[0])
