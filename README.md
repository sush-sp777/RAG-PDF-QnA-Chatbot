# 📄 Conversational Multi-PDF RAG Chatbot

A production-grade **Conversational RAG application** that allows users to upload multiple PDFs, chat with each document independently, and reliably handle **large PDFs** (50+ pages) without context leakage.
Built using **LangChain**, **Groq LLM (Llama 3.1-8B-instant)**, **Chroma Vector Store**, **HuggingFace embeddings**, and **Streamlit**.

---

## 🚀 Live Demo

Try the chatbot online on **Streamlit**:  
[🔗 Live Demo](https://lvfpemstezdoufsj4kviqc.streamlit.app/)

You can:

- Upload multiple PDFs
- Switch between documents
- Ask follow-up questions with document-scoped memory
- Reset chat per document

---

## 🧠 Problem Statement (Why this project matters)
Most beginner RAG apps fail when:
- Multiple PDFs are uploaded
- Chat history leaks between documents
- Large PDFs (>30–50 pages) return poor answers
- The same session memory is reused incorrectly
This project explicitly solves these real-world RAG problems.

---

## ✨ Key Features

1. ✅ Multi-PDF Support with Isolation
   - Each PDF gets its own vector store
   - No cross-document context leakage
2. ✅ Document-Scoped Conversational Memory
   - Chat history is bound to the selected PDF
   - You can switch PDFs and continue independent conversations
3. ✅ Dynamic Chunking for Large PDFs
   Chunk size and retrieval depth adapt automatically:
   - Small PDFs → smaller chunks
   - Large PDFs (50+ pages) → larger chunks + higher retrieval k
4. ✅ History-Aware Retrieval
   - Follow-up questions are reformulated into standalone queries
   - Improves accuracy for conversational flows
5. ✅ Page-Aware Answers
   - Answers reference PDF page numbers
   - Improves trust and explainability
6. ✅ Reset Chat Per Document
   - Clear chat history without re-embedding PDFs
     
---

## 🏗️ Architecture Overview
```
User → Streamlit UI
        ↓
PDF Upload → PyPDFLoader
        ↓
Text Splitter
        ↓
HuggingFace Embeddings
        ↓
Chroma Vector Store (per PDF)
        ↓
History-Aware Retriever
        ↓
Groq LLM (Llama 3.1-8B-instant)
        ↓
Answer with Page References
```
---

## 🔍 How It Works (High Level)

1. User uploads one or more PDFs
2. Each PDF is:
   - Loaded page-wise
   - Split using dynamic chunking
   - Embedded and stored in its own Chroma collection
3. User selects an active PDF
4. Questions are answered using:
   - PDF-specific vector store
   - PDF-specific chat history
5. Follow-ups are reformulated for better retrieval
   
---

## 🛠️ Tech Stack

- Streamlit – UI & state management
- LangChain – RAG chains, retrievers, memory
- Groq LLM – Llama 3.1-8B Instant
- HuggingFace Embeddings – all-MiniLM-L6-v2
- ChromaDB – Vector store
- Python-dotenv – Environment variable management

---

## ⚙️ Installation & Setup

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
1. Enter your Groq API key
2. Upload one or more PDFs
3. Select active PDF
4. Ask questions or summarize documents
5. Reset chat if needed
---
## 👨‍💻 Author

**Sushant Patil**

Generative AI Engineer

🔗 https://github.com/sush-sp777

🔗 https://www.linkedin.com/in/sushant-patil-9a05ab2a4/

---
