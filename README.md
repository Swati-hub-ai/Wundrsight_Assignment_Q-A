# Wundrsight_Assignment_Q-A

<img width="1920" height="1080" alt="2025-07-20 (6)" src="https://github.com/user-attachments/assets/3c9277d7-88c8-4575-a27e-6fcd398489b8" />


## Project Title: Medical RAG-based QA System using Groq API

This project was developed as part of the Wundrsight Software Engineering Intern assignment. The goal is to implement a **Retrieval-Augmented Generation (RAG)** system that can intelligently answer medical diagnostic questions based on an official document such as ICD or DSM in PDF format.

---

## Assignment Objective

> Build a simple RAG-based system that reads a PDF medical diagnostic guide and can answer medical questions using a local document and a hosted LLM.

---

##  Implemented

*  Accepts **PDF input** of medical classification systems (e.g., ICD-10, DSM)
*  Splits the documents into chunks using `RecursiveCharacterTextSplitter`
*  Converts chunks into dense embeddings using **MiniLM** from HuggingFace
*  Stores chunks in a **FAISS vector store**
*  Uses **Groq API** with **Mixtral-8x7B-Instruct** model for fast inference
*  Implements **RetrievalQA chain** to fetch top-k contexts and generate answers
*  Interactive UI using **Streamlit** with both:

  * Form-based input (`rag_qa_ui.py`)
  * Conversational memory-style chatbot (`rag_qa_ui_chat.py`)
*  Returns both **answer and source document context**

---

## 🗂 Folder Structure

```bash
project/
├── data/                          # Input folder for medical PDFs (e.g., ICD/DSM)
├── rag_qa_assignment_pdf.ipynb    # Assignment notebook with all core logic
├── rag_qa.py                   # Simple Notebook flow
├── rag_qa_ui_chat.py              # Streamlit Conversational Chatbot UI
├── sample_output.txt              # Required output for two mandatory questions
├── README.md                      # Assignment documentation
└── requirements.txt               # Python dependencies
```

---

## 🚀 Setup & Installation

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Create `.env` File

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3️⃣ Add Medical PDFs

Place any ICD or DSM PDFs inside the `data/` directory.

---

## 💻 How It Works

### ➤ Notebook Flow (`rag_qa.ipynb`):

1. Load PDF using `DirectoryLoader` + `PyPDFLoader`
2. Split document into \~400-character overlapping chunks
3. Embed chunks using `sentence-transformers/all-MiniLM-L6-v2`
4. Store embeddings in a FAISS vector store
5. Create `RetrievalQA` using Groq Mixtral model and top-3 context documents
6. Run queries and log responses with source context

### ➤ Required Sample Questions:

* “Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission”
* “What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?”

Answers and outputs saved in `sample_output.txt`

---

## 🖥️ UI Options

### 1. Conversational Chatbot UI (`rag_qa_ui_chat.py`)

```bash
streamlit run rag_qa_ui_chat.py
```
* Simple textbox for asking queries
* Displays top-3 context chunks with answers
* Chat interface with memory
* Groq LLM answers queries and remembers history
* Expander shows retrieved evidence chunks

---

## 📊 Sample Output Format

See `sample_output.txt` file.

```text
🔎 Query: Give me the correct coded classification for...
✅ Answer: F33.4
📌 Context Snippets: [Chunk 1...]

🔎 Query: What are the diagnostic criteria for OCD?
✅ Answer: Obsessional symptoms or compulsive acts...
📌 Context Snippets: [Chunk 2...]
```

---

## 📦 Dependencies

* Python 3.9+
* langchain
* langchain-community
* langchain-groq
* streamlit
* faiss-cpu
* sentence-transformers
* python-dotenv

Install via:

```bash
pip install -r requirements.txt
```

---

## Author

**Swati Kumari**
ISRO Intern | AI & ML Developer
[GitHub](https://github.com/) | [LinkedIn](https://www.linkedin.com/)

---

## ✅ Submission Checklist

*  PDF reader and chunker
*  Embedding and FAISS store
*  Groq-hosted LLM integration
*  Streamlit UI (2 modes)
*  Sample queries and output
*  Readme with full architecture

---

Feel free to reach out for questions or feedback!
