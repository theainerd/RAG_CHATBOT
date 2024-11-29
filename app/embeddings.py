import os
from io import BytesIO
import streamlit as st
from langchain_unstructured import UnstructuredLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App
st.title("PDF Embedding and Chroma DB Storage")
st.write("Upload your PDF files to create embeddings and store them in Chroma DB.")

# Upload files
uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    # Process Uploaded PDFs
    st.write("Processing uploaded PDFs...")
    docs = []

    # Create progress bar for file loading
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Loading {uploaded_file.name}..."):
            # Use UnstructuredLoader for file-like objects
            loader = UnstructuredLoader(file=BytesIO(uploaded_file.read()), metadata_filename=uploaded_file.name)
            docs.extend(loader.load())
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.write(f"Loaded {len(docs)} documents.")

    # Setup text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    # Create progress bar for splitting documents
    st.write("Splitting documents into chunks...")
    all_splits = []
    progress_bar = st.progress(0)
    for i, doc in enumerate(docs):
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(docs))

    st.write(f"Total chunks created: {len(all_splits)}")

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize Chroma
    vector_store = Chroma(embedding_function=embeddings)

    # Add documents to Chroma DB
    st.write("Adding documents to Chroma DB...")
    document_ids = []
    progress_bar = st.progress(0)
    for i, split in enumerate(all_splits):
        document_ids.append(vector_store.add_documents([split]))

        # Update progress bar
        progress_bar.progress((i + 1) / len(all_splits))

    st.success("Documents have been added to Chroma DB.")
    st.write(f"Total documents stored: {len(document_ids)}")
