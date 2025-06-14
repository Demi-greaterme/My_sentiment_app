# -*- coding: utf-8 -*-
"""Sentiment Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18TUbDeBY7FKcv-xagJXI-3y6O5uCiSqx
"""

import nltk
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import datasets
import joblib
from newspaper import Article
from transformers import pipeline
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

# Scraping sentences

URL = "https://www.theguardian.com/news/2025/jun/03/mrbeast-jimmy-donaldson-youtube-videos-star"

article = Article(URL)
article.download()
article.parse()
article.nlp()

text = article.text
print(text)

def process_raw_text_to_cleaned_sentences(raw_text_input):
    if not raw_text_input or not isinstance(raw_text_input, str):
        return []

    text_normalized_quotes = raw_text_input.replace("’", "'").replace("‘", "'").replace("`", "'")
    doc = nlp(text_normalized_quotes)

    cleaned_sentences = []
    for sent in doc.sents:
        sentence_text = sent.text
        cleaned = re.sub(r"[^a-zA-Z0-9\s\.,!'\"?\-()]", "", sentence_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = cleaned.lower()
        if cleaned: # Add only if not empty after cleaning
            cleaned_sentences.append(cleaned)
    return cleaned_sentences

#Getting rid of special characters
cleaned_sentences = process_raw_text_to_cleaned_sentences(text)
print(len(cleaned_sentences))
print(cleaned_sentences[:25])

# creating a dataset for the snetences
df = pd.DataFrame({'sentence': cleaned_sentences})

df.to_csv('labeled_sentences.csv', index=False)

df = pd.read_csv("labeled_sentences.csv")

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

df['label'] = df['sentence'].apply(lambda x: classifier(x)[0]['label'])

df.to_csv("labeled_sentences.csv", index=False)

df.head()

skewness = df['label'].value_counts()

sns.barplot(x=skewness.index, y=skewness.values)
plt.show()

# Handling categorical variables

label_map = {'label_0': 0, 'label_1': 1, 'label_2': 2}

df['label'] = df['label'].map(label_map)

dataset = Dataset.from_pandas(df)

# Tokenization + splitting
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['sentence'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

dataset_split = tokenized_dataset.train_test_split(test_size=0.2)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

model_path = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

demo_text = "I had so much fun yesterday!"

prediction = sentiment_pipeline(demo_text)
print(prediction)

import streamlit as st
from textblob import TextBlob

# Streamlit UI
st.title("Sentiment Analysis System")
st.subheader("Enter text below to analyze its sentiment")

# User input
user_input = st.text_area("Input Text", "Type here...")

if st.button("Analyze Sentiment"):
    if user_input:
        blob = TextBlob(user_input)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            sentiment = "Positive "
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Sentiment Score: {sentiment_score:.2f}")
    else:
        st.warning("Please enter text to analyze.")

with open('model.joblib', 'wb') as f:
    joblib.dump(model, f)

