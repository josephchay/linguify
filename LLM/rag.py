import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_answer(query):

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="qa_collection")

    query_embedding = model.encode([query])  # Convert user query to embedding

    # Retrieve top 3 similar results
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    if not results["metadatas"][0]:
        return "I couldn't find relevant information."

    # Prepare context from retrieved results
    context = "\n\n".join(
        [f"Q: {res['question']}\nA: {res['answer']}" for res in results["metadatas"][0]]
    )

    # Construct prompt for Ollama
    prompt = f"""
    You are an AI assistant. Answer the user's question based on the provided context. 
    If the answer is unclear or missing, respond with: "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """


    # Query Ollama
    response = ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]  # Return the generated response

# Example usage
query = "Explain RAG"
response = retrieve_answer(query)
print("Response from LLM:", response)
