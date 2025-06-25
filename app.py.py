# %%
# Importing necessary libraries
#!pip install newspaper3k
"""!pip install nltk
!pip install lxml_html_clean
!pip install spacy
!pip install transformers
!pip install scikit-learn
!pip install datasets
!pip install hf-xet
!pip install streamlit"""
import nltk
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import streamlit as st
from newspaper import Article
from transformers import pipeline
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from textblob import TextBlob
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

# %%
# Sentence scraping

URL = "https://www.theguardian.com/news/2025/jun/03/mrbeast-jimmy-donaldson-youtube-videos-star"

article = Article(URL)
article.download()
article.parse()
article.nlp()

text = article.text
print(text)

# %%
# splitting scraped text into individual sentences
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
        if cleaned:
            cleaned_sentences.append(cleaned)
    return cleaned_sentences

# %%
#Getting rid of special characters
cleaned_sentences = process_raw_text_to_cleaned_sentences(text)
print(len(cleaned_sentences))
print(cleaned_sentences[:5])

# creating a pandas dataframe for the sentences
df = pd.DataFrame({'sentence': cleaned_sentences})

df['label'] = None

df.to_csv("cleaned_sentences.csv", index = False)

# %%
#Auto-classification with roBERTa
classifier = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model = classifier, tokenizer = classifier)

pd.read_csv("cleaned_sentences.csv")
label_mapping = {"LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
                }
df["label"] = df["sentence"].apply(
    lambda x: label_mapping[sentiment_pipeline(x)[0]["label"]]
)

# %%
df.to_csv("auto_labeled.csv", index=False)
DF = pd.read_csv("auto_labeled.csv")
print(DF.head())

# %%

# Tokenization
dataset = Dataset.from_pandas(DF)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# %%
# downgrading to compatible library
#!pip install numpy==1.24.4

#Freezing and fine-tuning
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
for parameter in model.distilbert.transformer.layer[:3].parameters():
    parameter.requires_grad = False

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    weight_decay=0.01)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset_split = tokenized_dataset.train_test_split(test_size=0.2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer,
    data_collator=data_collator)

trainer.train()
trainer.evaluate()
trainer.save_model("./finetuned_distilbert")

# %%
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "I'm very impressed with the quality of this product!"
result = sentiment_pipeline(text)
print(result)

# %%
# Deployment
st.title("Sentiment Analysis System")
st.subheader("Enter text below to analyze its sentiment")
user_input = st.text_area("Input Text", "Type here...")

if st.button("Analyze Sentiment"):
    if user_input:
        blob = TextBlob(user_input)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Sentiment Score: {sentiment_score:.2f}")
    else:
        st.warning("Please enter text to analyze.")

# %%
with open('model.joblib', 'wb') as f:
    joblib.dump(model, f)

# %%



