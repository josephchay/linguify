import json
import chromadb
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSON data
with open("data.json", "r") as f:
    data = json.load(f)

# Convert Q&A into embeddings
documents = [f"Q: {item['question']} A: {item['answer']}" for item in data]
embeddings = model.encode(documents)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="qa_collection")

# Store embeddings in ChromaDB
for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding.tolist()],
        metadatas=[{"question": data[idx]["question"], "answer": data[idx]["answer"]}]
    )

print("Data stored in ChromaDB.")
