# Linguify Self-Reflective (SR)-RAG Agent Architecture System

Our novel Self-Reflective (SR)-RAG enables better instruction-tuning and Query-routing for our Large Language Model (LLM) system designed to generate human-like responses to text prompts. This repository provides the code and necessary steps for setting up and using the LinguifyChat LLM agent.

## Setup & Development

Follow the instructions below to set up the LinguifyChat LLM system on your local environment.

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/josephchay/linguify.git
```

```bash
cd linguify/LLM
```

### 2. Install Dependencies

After cloning the repository, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries and packages for the project.

## Workflow

The LinguifyChat system consists of three main stages: scraping, indexing, and testing the RAG pipeline. Follow these steps sequentially to run the system.

### 1. Scrape the Website

Run the `lhdn-scrapper.ipynb` notebook to scrape the relevant data from the target website. This will collect the information needed for the LLM system.

To run the notebook, use the following command:

```bash
jupyter notebook lhdn-scrapper.ipynb
```

### 2. Run the Indexing Process

Once the data is scraped, run the `indexing.ipynb` notebook to index the scraped data. This process prepares the data for use in the LLM system.

```bash
jupyter notebook indexing.ipynb
```

### 3. Test the RAG Pipeline

Finally, test the Retrieval-Augmented Generation (RAG) pipeline by running the `llm.ipynb` notebook. This will validate that the system is working as expected.

```bash
jupyter notebook llm.ipynb
```

## Changelog

Refer to the [CHANGELOG](CHANGELOG.md) file for detailed updates, changes, and new features added to the LinguifyChat system.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
