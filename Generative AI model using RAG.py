import torch
import numpy as np
import logging
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Bio import Entrez

logging.basicConfig(level=logging.INFO)
Entrez.email = "gayatrirayasam@gmail.com"

def fetch_pubmed_articles(num_articles=5):
    handle = Entrez.esearch(db="pubmed", term="machine learning", retmax=num_articles)
    record = Entrez.read(handle)
    handle.close()
    ids = record['IdList']
    
    articles = []
    for pmid in ids:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        article = Entrez.read(handle)
        handle.close()
        title = article[0]['MedlineCitation']['Article']['ArticleTitle']
        abstract = article[0]['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [''])[0]
        articles.append(f"{title}. {abstract}")
    return articles

documents = fetch_pubmed_articles(num_articles=5)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=False)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

document_embeddings = embedding_model.encode(documents)

def retrieve_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def generate_answer(question):
    try:
        retrieved_docs = retrieve_documents(question)
        inputs = tokenizer(question, return_tensors="pt")
        context_input_ids = tokenizer(retrieved_docs, return_tensors="pt", padding=True, truncation=True).input_ids
        generated_ids = model.generate(input_ids=inputs['input_ids'], 
                                       attention_mask=inputs['attention_mask'], 
                                       context_input_ids=context_input_ids)
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer."

def batch_generate_answers(questions):
    answers = {}
    for question in questions:
        answers[question] = generate_answer(question)
    return answers

questions = [
    "What is machine learning?",
    "Can you explain the significance of PubMed?",
    "What are the applications of machine learning in healthcare?",
    "Tell me about recent advancements in machine learning."
]

answers = batch_generate_answers(questions)
for question, answer in answers.items():
    print(f"Question: {question}\nAnswer: {answer}\n")
