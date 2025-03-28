import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_answer(query):
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="LLM/finance/e-invoice_db")
    collection = chroma_client.get_or_create_collection(name="finance_pdfs")

    # Convert user query to embedding
    query_embedding = model.encode([query])

    # Retrieve top 3 most relevant text chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    # Extract the retrieved document texts
    retrieved_texts = results["documents"][0] if results["documents"] else []

    if not retrieved_texts:
        return "I couldn't find relevant information."

    # Prepare context from retrieved results
    context = "\n\n".join(retrieved_texts)

    # Construct prompt for Ollama
    prompt = f"""
    You are an AI assistant specializing in finance-related topics. 
    Answer the user's question based on the provided context.
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
query = "Can i request for e-invoice after i bought an air-ticket"
response = retrieve_answer(query)
print("Response from LLM:", response)
