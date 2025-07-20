
# Conversational RAG QA chatbot UI using Groq API and Streamlit

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up Streamlit page
st.set_page_config(page_title="🧠 Medical Chatbot - RAG (Groq)", layout="wide")
st.title("💬 Wundrsight Conversational Medical Chatbot (Groq)")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load PDFs and build vector store
@st.cache_resource
def load_vectorstore():
    loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding_model)

vectorstore = load_vectorstore()

# Load Groq LLM
llm = ChatGroq(
    model_name="gemma2-9b-it",
    api_key=GROQ_API_KEY
)

# Prompt template and QA chain
prompt_template = PromptTemplate(
    template="""
Use the information in the context below to answer the question.
If unsure, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Type your medical question...")

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({"query": user_query})
            response = result["result"]
            st.markdown(response)

            # Optional: show sources
            with st.expander("📌 Show retrieved context"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}...")

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})
