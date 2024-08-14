# Manufacturing Query System - RAG Document Search AI/ML

## Overview

The Manufacturing Query System is a Retrieval-Augmented Generation (RAG) AI/ML-powered document search tool designed specifically for the manufacturing industry. The system leverages state-of-the-art Natural Language Processing (NLP) models to provide accurate and contextually relevant answers to queries by searching and analyzing a large corpus of manufacturing-related documents.

## Features

- **Natural Language Querying**: Users can ask questions in natural language, and the system will return the most relevant documents or specific answers.
- **Contextual Understanding**: The system utilizes advanced AI models to understand the context of queries, ensuring that the responses are accurate and relevant.
- **Streamlit Interface**: A user-friendly web interface powered by Streamlit, allowing for easy interaction with the system.

## Getting Started

### Prerequisites

Before running the system, ensure you have the following:

- Python 3.11 or higher installed on your system.
- API keys and tokens:
  - **OpenAI API Key**: Required for accessing GPT-based models.
  - **Hugging Face Token**: Required for accessing pre-trained models on Hugging Face.

### Installation

Set Up Environment Variables

Create a .env file in the root directory of the project and add the following environment variables:

```bash
   OPENAI_API_KEY=your_openai_api_key_here
   HF_TOKEN=your_hugging_face_token_here
```

Replace your_openai_api_key_here and your_hugging_face_token_here with your actual API key and token.

Run the following command to install all necessary Python packages:

```bash
    pip install -r requirements.txt
    apt-get install poppler-utils
    apt-get install tesseract-ocr
    pip install pytesseract
    pip install langchain_openai
```

Running the System
After setting up the environment variables and installing the dependencies, you can start the system by running the following command:

```bash
    streamlit run main.py
```

# Docker Setup for a Streamlit Application

This guide will walk you through building a Docker image for a Streamlit application.

## Step 1: Create a `Dockerfile`

Create a `Dockerfile` in your project directory with the following content:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY app/requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY app/ .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
```


```bash
    docker build -t my-python-app .
```