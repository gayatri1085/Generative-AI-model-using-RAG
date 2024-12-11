import torch
import numpy as np
import logging
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=False)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is a series of fortifications made of various materials.",
    "The Statue of Liberty was a gift from France to the United States.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."
]

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
    "Where is the Eiffel Tower located?",
    "What is the Great Wall of China?",
    "Tell me about Python.",
    "What is machine learning?"
]

answers = batch_generate_answers(questions)
for question, answer in answers.items():
    print(f"Question: {question}\nAnswer: {answer}\n")