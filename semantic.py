import json
import hnswlib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load data from JSON file
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    questions = [item['question'] for item in data['questions']]
    answers = [item['answer'] for item in data['questions']]
    return questions, answers

# Generate embeddings using Hugging Face transformers
def generate_embeddings(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over the token embeddings
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)


questions, answers = load_data("Ecommerce_FAQ_Chatbot_dataset.json")

# Generate embeddings
question_embeddings = generate_embeddings(questions)

# Build HNSW index
dim = question_embeddings.shape[1] 
index = hnswlib.Index(space='cosine', dim=dim)  
index.init_index(max_elements=len(question_embeddings), ef_construction=200, M=48) 
index.add_items(question_embeddings)

index.set_ef(50)  

# Perform semantic search
def semantic_search(query, k=5):
    query_embedding = generate_embeddings([query])
    labels, distances = index.knn_query(query_embedding, k=k) 
    results = [(questions[label], answers[label]) for label in labels[0]]
    return results

# Test query
query = "How do i register?"
results = semantic_search(query, k=3)  # Retrieve top 3 matches
print(f"Query: {query}\nResults:")
for i, (question, answer) in enumerate(results):
    print(f"{i+1}. Question: {question}\n   Answer: {answer}")

  
