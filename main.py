import streamlit as st
import nltk
from gtts import gTTS
import os
from io import BytesIO

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag, ne_chunk, word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from nltk.util import ngrams
import re
import requests
# Function to download NLTK resources if missing
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker_tab')

    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')

# Ensure resources are available at the start of the app
download_nltk_resources()

# NLP Functions
def lemmatize_sentence(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def tokenize_text(text):
    return word_tokenize(text)

def stem_words(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    return ' '.join([ps.stem(word) for word in words])

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])

def pos_tagging(text):
    words = word_tokenize(text)
    return pos_tag(words)

def named_entity_recognition(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return ne_chunk(pos_tags)

def normalize_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns sentiment score (-1 to 1)

def generate_ngrams(text, n):
    words = word_tokenize(text)
    return list(ngrams(words, n))

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    return X, vectorizer.get_feature_names_out()

def summarize_text(text, max_length=30, min_length=10):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# New: Text-to-Speech Function
def text_to_speech(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)  # Save to BytesIO buffer
    audio_buffer.seek(0)  # Reset buffer position
    return audio_buffer

# Streamlit UI
st.title("NLP Functions Project")

# Input text
input_text = st.text_area("Enter text here", height=200)

# Buttons to trigger each function
if st.button('Lemmatize'):
    lemmatized_text = lemmatize_sentence(input_text)
    st.write("Lemmatized Text:", lemmatized_text)

if st.button('Tokenize'):
    tokens = tokenize_text(input_text)
    st.write("Tokens:", tokens)

if st.button('Stem Words'):
    stemmed_text = stem_words(input_text)
    st.write("Stemmed Words:", stemmed_text)

if st.button('Remove Stopwords'):
    filtered_text = remove_stopwords(input_text)
    st.write("Text without Stopwords:", filtered_text)

if st.button('POS Tagging'):
    pos_tags = pos_tagging(input_text)
    st.write("POS Tags:", pos_tags)

if st.button('NER (Named Entity Recognition)'):
    entities = named_entity_recognition(input_text)
    st.write("Named Entities:", entities)

if st.button('Normalize Text'):
    normalized_text = normalize_text(input_text)
    st.write("Normalized Text:", normalized_text)

if st.button('Sentiment Analysis'):
    sentiment_score = sentiment_analysis(input_text)
    st.write("Sentiment Score:", sentiment_score)

if st.button('Generate N-grams'):
    n = st.slider("Select N for N-grams", min_value=1, max_value=5, value=2)
    ngrams_generated = generate_ngrams(input_text, n)
    st.write(f"{n}-grams:", ngrams_generated)

if st.button('TF-IDF'):
    tfidf_matrix, feature_names = compute_tfidf([input_text])
    st.write("TF-IDF Matrix:\n", tfidf_matrix.toarray())
    st.write("Feature Names:", feature_names)

if st.button('Summarize Text'):
    summary = summarize_text(input_text)
    st.write("Summary:", summary)


# Text-to-Speech Button
if st.button('Convert Text to Speech'):
    tts_audio = text_to_speech(input_text)
    st.audio(tts_audio, format='audio/mp3')