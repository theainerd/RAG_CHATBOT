from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

from langchain_core.prompts import PromptTemplate

template = """You are a skin care treatment specialist who answer treatment procedure
Use the following pieces of context to answer the question at the end, don't try to make up an answer.
Answer strictly from context else say I dont know.

{context}

#### sample treatment ####

Standard Manicure: 40 minutes
Cost: $58
Preparation:
1. 1 x small towel draped across the table.
2. 1 x small towel rolled for hand to rest on.
3. 1 x A4 plastic square to rest hand.
4. Cuticle Oil.
5. Base Coat
6. Top Coat
7. Nail clippers.
8. Nail file.
9. Block buffer.
10. Cuticle pusher to clean under nails.
11. Nail Polish remover + cotton balls.
Client Preparation:
1. Greet client and confirm polish colour.
2. Ask if they would like a tea or coffee.
3. Once the client is seated, go through the consultation form with them. Make
note, of any requests, concerns or contra-indications they may have and address
these.
4. Check nails for contra-indications. If unsure of a condition, ask client to seek
medical advice. Always check with management before informing client.
5. If client is wearing nail polish, remove polish from nails.

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# Setup
llm = ChatOpenAI(model="gpt-4o-mini")

# Load existing vector store (Chroma) from a directory
vector_store_directory = "chroma"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load the existing vector store from the directory
vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_directory)
retrieved_docs = vector_store.similarity_search(state["question"])

# Define the state for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    # Perform similarity search in the loaded vector store
    retrieved_docs = vector_store.similarity_search(state["question"])
    print("retrieved docs:",retrieved_docs)
    return {"context": retrieved_docs}

def generate(state: State):
    # Combine the content of the retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Invoke the prompt with the question and context
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Test the QA process
response = graph.invoke({"question": "tell me treatment for redness, sun damage, fine lines & wrinkles"})
print(response["answer"])
