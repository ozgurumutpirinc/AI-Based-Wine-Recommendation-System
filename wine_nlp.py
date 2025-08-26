import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
"""
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
"""
from MissingValues import missing_values_table

# EDA & EVALUATING MISSING VALUES
df = pd.read_csv('wine.csv')
df.head()
df.info()
df.isna().sum()

missing_values_table(df)

df.shape

df = df.drop(['Unnamed: 0', 'region_2', 'region_1', 'designation', 'taster_twitter_handle', 'taster_name'], axis = 1)
df.head()

df['price'].fillna(df['price'].median(), inplace = True)
df['country'].fillna(df['country'].mode()[0], inplace = True)
df['province'].fillna(df['province'].mode()[0], inplace = True)

df = df.dropna()

# TEXT CLEANING
from text_cleaner import clean_text_spacy, lemmatize_texts
df['cleaned_description'] = df['description'].apply(clean_text_spacy)
df['cleaned_title'] = df['title'].apply(clean_text_spacy)

df['cleaned_description'] = lemmatize_texts(df['cleaned_description'].tolist())
df['cleaned_title'] = lemmatize_texts(df['cleaned_title'].tolist())

def prepare_text(row):
    return f"Title: {row['title']} | Description: {row['description']}"

df["Text Data"] = df.apply(prepare_text, axis=1)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(
        page_content =  row['Text Data'],
        metadata = {
            "country": row['country'],
            "points": row['points'],
            "price": row['price'],
            "province": row['province'],
            "variety": row['variety'],
            "winery": row['winery']
        }
    )
    for _, row in df.iterrows()
]


vectordb = Chroma.from_documents(documents = documents, embedding = model, persist_directory = './WineDB')
vectordb.persist()
