# Conversational RAG Chatbot with PDF Uploads and Chat History

This project is a **Conversational AI application** that allows users to upload PDF documents and interact with their content via a chat interface. It uses **LangChain**, **Groq LLM (Llama 3.1-8B)**, **Chroma Vector Store**, and **HuggingFace embeddings** for retrieval-augmented generation (RAG).

---

## Features

- Upload multiple PDF documents.  
- Split PDFs into chunks and generate embeddings.  
- Store embeddings in a vector store (Chroma) for fast retrieval.  
- Chat with uploaded documents using Groq LLM.  
- Keep **session-based chat history**.  
- Reformulate questions to be **standalone** for better context.  

---

## Tech Stack

- **Streamlit** - Frontend interface  
- **LangChain** - RAG pipelines, prompts, chains, message history  
- **Groq LLM** - Llama 3.1-8B Instant  
- **HuggingFace Embeddings** - sentence-transformers/all-MiniLM-L6-v2  
- **Chroma** - Vector storage for document embeddings  
- **Python-dotenv** - Load API keys securely  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sush-sp777/RAG-PDF-QnA-Chatbot.git
cd RAG-PDF_OnA-Chatbot
```
2. Create a virtual environment and activate:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a .env file in the project root and add:
```bash
HUGGINGFACE_TOKEN=your_huggingface_token
```
---
## Usage

1. Run Streamlit app:
```bash
streamlit run app.py
```
2. Enter your Groq API Key in the text input.
3. Optionally, choose or enter a Session ID.
4. Upload PDF documents.
5. Ask questions in the chat input.
6. Chat history is stored per session.
---
