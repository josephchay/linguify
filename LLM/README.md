# LinguifyChat LLM Agent Architecture System

LinguifyChat is a Large Language Model (LLM) system designed to generate human-like responses to text prompts, with specialized knowledge in financial and tax domains. This repository provides the code and necessary steps for setting up and using the LinguifyChat LLM agent.

For comprehensive information about our implementation details, methodology, and findings, please refer to our [research documentation](./research_documentation.md) which thoroughly documents everything we've accomplished in this codebase.

## Architecture Overview

LinguifyChat uses a Self-Reflective Retrieval-Augmented Generation (SR-RAG) architecture that enhances traditional RAG systems with a self-reflection mechanism. This allows the model to:

1. Retrieve relevant information from specialized knowledge bases
2. Generate high-quality responses based on the retrieved information
3. Reflect on and improve its responses before finalizing

### Key Components

- **Document Scraping**: Automated extraction of domain-specific knowledge
- **Vector Database**: Efficient storage and retrieval of knowledge chunks
- **Embedding Model**: Semantic understanding of queries and documents
- **LLM Core**: Response generation with self-reflection capabilities

## Setup & Development

Follow the instructions below to set up the LinguifyChat LLM system on your local environment.

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```
git clone https://github.com/josephchay/linguify.git
cd linguify/LLM
```

### 2. Install Dependencies

After cloning the repository, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries and packages for the project, including:

- langchain
- chromadb
- huggingface_hub
- sentence-transformers
- torch
- requests
- beautifulsoup4
- pymupdf

## Data Collection & Preprocessing

LinguifyChat is trained on a comprehensive dataset that includes financial and tax information. The following steps explain how to collect and preprocess this data.

### 1. Scrape Website Data

Run the `lhdn-scraper.ipynb` notebook to collect data from financial and tax websites:

```bash
jupyter notebook lhdn-scraper.ipynb
```

The scraper extracts text from websites and preserves the structure, including tables which are converted to markdown format. Example usage:

```python
# Define target website
root_url = "https://www.hasil.gov.my/en/individual/"
# Run the scraper
results = scrape_all_individual_pages(root_url)
# Save results to file
with open("tax-data.txt", "w", encoding="utf-8") as f:
    for url, content in results.items():
        f.write(f"\n--- {url} ---\n{content}\n")
```

### 2. Process PDF Documents (Optional)

For PDF documents containing financial information, use the PDF extraction utilities:

```python
from utils.pdf_extractor import extract_text_from_pdf

# Extract text from a single PDF
pdf_text = extract_text_from_pdf("path/to/document.pdf")

# Or process an entire directory
documents = extract_all_pdf_content("path/to/pdf/directory")
```

## Indexing & Vector Database

After collecting the data, the next step is to create a vector database for efficient retrieval.

### 1. Run the Indexing Process

Execute the `indexing.ipynb` notebook to create embeddings and store them in a ChromaDB:

```bash
jupyter notebook LLM___indexing.ipynb
```

Custom indexing can also be performed via code:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Create and store embeddings
client = chromadb.PersistentClient(path="./vector-db/tax_knowledge")
collection = client.get_or_create_collection(name="tax_information")

# Add documents to the collection
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
embedded_chunks = embeddings.embed_documents(texts)
ids = [str(i) for i in range(len(texts))]

collection.add(
    ids=ids,
    embeddings=embedded_chunks,
    documents=texts,
    metadatas=metadatas
)
```

## Using the LinguifyChat System

Once the data is indexed, you can use the LinguifyChat system to answer queries.

### 1. Simple Query Interface

```python
from linguify.llm import LinguifyChat

# Initialize the chat system
chat = LinguifyChat(
    vector_db_path="./vector-db/tax_knowledge",
    collection_name="tax_information"
)

# Ask a question
response = chat.ask("How do I calculate my income tax in Malaysia?")
print(response)
```

### 2. Advanced Usage with Self-Reflection

```python
from linguify.llm import SelfReflectiveRAG

# Initialize the advanced RAG system
rag = SelfReflectiveRAG(
    vector_db_path="./vector-db/tax_knowledge",
    collection_name="tax_information",
    reflection_threshold=0.7
)

# Ask a complex question with self-reflection
response = rag.query_with_reflection(
    "What are the tax implications when selling a property in Malaysia?",
    max_reflection_rounds=2
)

print(f"Final answer: {response['answer']}")
print(f"Reflection process: {response['reflection_log']}")
```

### 3. Batch Processing

For handling multiple queries efficiently:

```python
queries = [
    "What are the income tax brackets in Malaysia?",
    "How do I register as a taxpayer?",
    "What deductions are available for individuals?"
]

results = chat.batch_process(queries, max_workers=3)
for query, response in zip(queries, results):
    print(f"Q: {query}\nA: {response}\n")
```

## Troubleshooting

If you encounter issues with the embedding model:

```python
# Test the embedding model directly
from langchain.embeddings import HuggingFaceEmbeddings

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_embedding = embeddings.embed_query("Test query")
    print(f"Embedding dimension: {len(test_embedding)}")
    print("Embedding model is working correctly")
except Exception as e:
    print(f"Embedding model error: {e}")
    print("Try reinstalling sentence-transformers with: pip install -U sentence-transformers")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LinguifyChat in your research or application, please cite:

```
@software{linguifychat2025,
  author = {Chay, Joseph and Leong, TinJet},
  title = {LinguifyChat: Self-Reflective RAG for Domain-Specific Conversations},
  year = {2025},
  url = {https://github.com/josephchay/linguify}
}
```

## Contact

For questions, suggestions, or issues related to this dataset, please contact the creators through [Hugging Face](https://huggingface.co/josephchay/Linguify) or open an issue in our GitHub [repository](https://github.com/josephchay/linguify).
