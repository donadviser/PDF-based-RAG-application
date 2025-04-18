# PDF-based-RAG-application
A Python-based Retrieval-Augmented Generation (RAG) application for querying PDF documents. This project demonstrates advanced RAG techniques, including local LLM integration, vector database updates, and response quality testing.



# Using OllamaEmbeddings on a Local PC

This guide provides step-by-step instructions to set up and use `OllamaEmbeddings` from LangChain to generate text embeddings locally on your PC. This is useful for Retrieval-Augmented Generation (RAG) applications, such as indexing and querying documents in a vector database.

## Installation and Setup

### 1. Install Ollama
- Visit [ollama.com](https://ollama.com/) and download the installer for your operating system.
- Run the installer and follow the prompts to set up Ollama.

### 2. Start the Ollama Server
- Open a terminal or command prompt.
- Run the following command to start the Ollama server:
  ```bash
  ollama serve
  ```
- This launches a local REST API server at `http://localhost:11434`. Keep the terminal open while using Ollama.

### 3. Pull an Embedding Model
- Choose a lightweight embedding model, such as `nomic-embed-text`.
- Download the model by running:
  ```bash
  ollama pull nomic-embed-text
  ```
- Verify the model is available:
  ```bash
  ollama list
  ```

### 4. Install Python Dependencies
- Install the required Python packages using pip:
  ```bash
  pip install langchain ollama
  ```
- These packages provide the `OllamaEmbeddings` module and ensure compatibility with the Ollama server.

## Using OllamaEmbeddings in Python

### 5. Configure OllamaEmbeddings
- Create a Python script (e.g., `embeddings.py`) and add the following code to initialize the embedding function:
  ```python
  from langchain.embeddings import OllamaEmbeddings

  # Initialize the embedding function
  embedding_function = OllamaEmbeddings(model="nomic-embed-text")
  ```