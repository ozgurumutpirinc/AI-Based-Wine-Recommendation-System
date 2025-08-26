import spacy
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import string

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text_spacy(text):
    """HTML temizleme, lowercase ve sadece harf bırakma."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

def lemmatize_texts(texts, batch_size=500, n_process=1):
    """Toplu tokenization + lemmatization + stopwords temizleme"""
    cleaned_texts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts)):
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts
