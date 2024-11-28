import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Setup
pdf_directory = "data/"
loader = PyPDFDirectoryLoader(pdf_directory)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(embedding_function=embeddings)

# Load documents from the directory
print("Loading PDFs...")
docs = loader.load()
print(docs)
print(f"Loaded {len(docs)} documents.")

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

# Split documents into chunks and add to the vector store
print("Splitting documents into chunks...")
all_splits = []
for doc in tqdm(docs, desc="Splitting Documents", unit="doc"):
    splits = text_splitter.split_documents([doc])  # Split each document
    all_splits.extend(splits)

print(all_splits)

# Add split documents to the vector store
print("Adding documents to vector store...")
document_ids = []
for split in tqdm(all_splits, desc="Adding to Vector Store", unit="split"):
    document_ids.append(vector_store.add_documents([split]))

print("Vector store created and documents added.")
