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