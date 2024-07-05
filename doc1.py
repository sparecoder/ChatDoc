import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Vectorstore Db
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector embedding
from dotenv import load_dotenv
import tempfile


load_dotenv()
# Load the GROQ and the Google API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("ChatDoc")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
You are an experienced and empathetic medical assistant. Analyze the following patient file and provide accurate responses and suggestions based on their health condition.

Patient File:
{context}

Questions:
{input}

If specific information is not available in the patient file, use your medical knowledge and provide general guidelines based on the patient's condition.
"""
)


def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Attempt to load the PDF document directly
        try:
            loader = PyPDFLoader(file_path)
            st.session_state.docs = loader.load()  # Document loading
            
            # Debug: Check if documents are loaded
            if not st.session_state.docs:
                st.error("No documents loaded. Please check the uploaded file.")
                return
            st.write(f"Loaded {len(st.session_state.docs)} documents")
            
            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            
            # Debug: Check if documents are split correctly
            if not st.session_state.final_documents:
                st.error("Document splitting failed. Please check the text splitter configuration.")
                return
            st.write(f"Split into {len(st.session_state.final_documents)} chunks")
            
            # Generate embeddings for the split documents
            try:
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return
            
            # Debug: Check if embeddings are generated
            embeddings = st.session_state.embeddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])
            if not embeddings or len(embeddings[0]) == 0:
                st.error("Embedding generation failed. Please check the embeddings configuration.")
                return
            st.write(f"Generated embeddings for {len(embeddings)} chunks")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name
        st.session_state.uploaded_file_path = temp_file_path
        
        # Debug: Ensure the file is saved correctly
        st.write(f"Uploaded file saved to: {temp_file_path}")

prompt1 = st.text_input("What do you want to ask from ChatDoc")

if st.button("Creating Vector Store"):
    vector_embedding(st.session_state.uploaded_file_path)
    st.write("Vector Store DB is ready")

import time

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-----------------------------------")
    else:
        st.error("Vector Store is not ready. Please create the vector store first.")
