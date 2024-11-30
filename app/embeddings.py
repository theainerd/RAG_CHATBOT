import os
from io import BytesIO
import streamlit as st
from langchain_unstructured import UnstructuredLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App Setup
st.set_page_config(page_title="Skincare Treatment Assistant", layout="wide")
st.title("💆 Skincare Treatment Assistant")
st.write("Upload PDFs to create embeddings, store them in Chroma DB, and interact using a chat interface.")

# Directory for persistent storage
vector_store_directory = "chroma"

# Initialize Chroma with persistence
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_directory)

# Define the Prompt Template
template = """You are a skincare treatment specialist who answers treatment procedure questions.
Use the following pieces of context to answer the question at the end.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# Setup the Chat Interface
llm = ChatOpenAI(model="gpt-4o-mini")

# File Upload and Embedding Process
uploaded_files = st.file_uploader("📂 Upload PDF Files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    st.write("Processing uploaded PDFs...")
    docs = []

    # Process Uploaded PDFs
    for uploaded_file in uploaded_files:
        with st.spinner(f"Loading {uploaded_file.name}..."):
            loader = UnstructuredLoader(file=BytesIO(uploaded_file.read()), metadata_filename=uploaded_file.name)
            docs.extend(loader.load())

    st.write(f"Loaded {len(docs)} documents.")

    # Text Splitting and Embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)

    # Add documents to Chroma DB
    for split in all_splits:
        vector_store.add_documents([split])

    # Persist the database
    vector_store.persist()
    st.success("📚 Documents have been added to Chroma DB.")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user query
if prompt := st.chat_input("Ask a question about skincare treatments:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Processing your question..."):
        # Retrieve documents from the vector store
        retrieved_docs = vector_store.similarity_search(prompt, k=5)
        if not retrieved_docs:
            docs_content = "I couldn't find relevant information in the database. Please answer the question to the best of your knowledge."
        else:
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content"))

        # Prepare the prompt input
        prompt_input = {"context": docs_content, "question": prompt}

        # Generate a response using the LLM
        response = llm.invoke(prompt.format(**prompt_input))

        # Extract the response content
        response_text = response.content if hasattr(response, "content") else "I'm sorry, I couldn't process your question."

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

