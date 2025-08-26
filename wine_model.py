from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
import shutil
shutil.rmtree("WineDB", ignore_errors=True)
vectordb = Chroma(
    persist_directory="WineDB", 
    embedding_function=model
)

def short(text, max_chars=200):
    if not text:
        return ""
    text = text.strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text

llm_model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

def generate_recommendation(query, k = 3):
    
    results = vectordb.similarity_search(query, k = k)

    context = "\n".join([
    f"""
    Wine: {doc.metadata.get('variety', 'Unknown')}
    Country: {doc.metadata.get('country', 'Unknown')}
    Points: {doc.metadata.get('points', 'Unknown')}
    Price: {doc.metadata.get('price', 'Unknown')}
    Winery: {doc.metadata.get('winery', 'Unknown')}
    Notes: {short(doc.page_content, 250)}
    """
    for doc in results
    ])

    prompt = f"""
    You are a wine expert. 
    User request: {query}

    Based on the context below, recommend 3 wines. 
    - Each recommendation should be written in a full, natural sentence.
    - Include: variety, country, approximate price, and one short tasting note.
    - Only use information from the context. Do NOT repeat instructions or list items mechanically.

    Context:
    {context}

    Answer:
    """.strip()

    inputs = tokenizer(prompt, return_tensors = "pt", truncation = True, max_length = 1024)
    outputs = llm_model.generate(**inputs, max_new_tokens = 250, do_sample = True, top_k = 50, top_p = 0.95, temperature = 0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

