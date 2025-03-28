import fitz  # PyMuPDF
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_all_pdf_content(directory):
    """Extract text from all PDFs in a directory and return as a list."""
    all_text = []
    for file_name in os.listdir(directory):  
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            pdf_text = extract_text_from_pdf(file_path)
            print(f"Extracted content from: {file_name}\n")
            print(pdf_text[:500])  # Preview first 500 characters
            all_text.append(pdf_text)

    return all_text  # Return list of extracted texts

def split_text(texts):
    """Split extracted text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def store_in_vector_db(chunks):
    """Store text chunks in a vector database using Sentence Transformers embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded_chunks = embeddings.embed_documents(chunks)  # Generate embeddings

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="LLM/finance/e-invoice_db")  # Saves DB locally
    collection = client.get_or_create_collection(name="finance_pdfs")

    # Add documents to ChromaDB
    for i, (text, embedding) in enumerate(zip(chunks, embedded_chunks)):
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[text]
        )

    print("Stored in vector database successfully!")
    return client

def save_to_txt(text_list, output_file="extracted_text.txt"):
    """Save extracted text list into a file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_list))  # Join list into a single string
    
    print(f"Extracted text saved to {output_file}")

if __name__ == "__main__":
    pdf_texts = extract_all_pdf_content("LLM/finance/E-invoice-pdf")  # Extract all PDFs
    chunks = split_text(pdf_texts)  # Split into chunks
    save_to_txt(chunks)
    db = store_in_vector_db(chunks)  # Store in vector database
