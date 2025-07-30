# import os
# import io
# import requests
# import pypdf
# import asyncio
# import warnings
# warnings.simplefilter(action='ignore')
# from typing import List
# from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv

# # LangChain Imports
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# # --- Configuration ---
# PERSIST_DIRECTORY = "./chroma_db"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # "nvidia/embed-qa-4" or "sentence-transformers/all-MiniLM-L6-v2"
# LLM_PROVIDER = "nvidia" # Choose your LLM provider: 'google' or 'openai' or 'nvidia'
# GOOGLE_MODEL_NAME = "gemini-1.5-flash" # For Google: "gemini-1.5-flash", "gemini-pro"
# OPENAI_MODEL_NAME = "gpt-3.5-turbo" # For OpenAI: "gpt-4o", "gpt-3.5-turbo"
# NVIDIA_MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1" # "mistralai/mixtral-8x7b-instruct-v0.1" or "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

# # --- Load environment variables ---
# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="LLM-Powered Intelligent Query-Retrieval System",
#     description="API for processing policy documents and answering questions using RAG.",
#     version="1.0.0",
# )

# # --- ADDED: Security Configuration ---
# # Define the security scheme and the expected token from the project spec
# auth_scheme = HTTPBearer()
# EXPECTED_TOKEN = "5639d6715d8cfe974e313a8fe74f2394761238a54ba9d15d550d11a8e5a767ee"

# def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
#     """Dependency function to verify the Bearer token."""
#     if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
#         raise HTTPException(
#             status_code=403,
#             detail="Forbidden: Invalid authorization token",
#         )
#     return credentials.credentials

# # --- Global variables for the RAG components ---
# vector_store = None
# rag_chain = None

# # --- Pydantic Models for API ---
# class QueryRequest(BaseModel):
#     documents: List[str] = Field(..., description="List of document URLs to query (currently supports one).")
#     questions: List[str] = Field(..., description="List of natural language questions to answer.")

# class Answer(BaseModel):
#     answer: str

# class QueryResponse(BaseModel):
#     answers: List[Answer]

# class IngestRequest(BaseModel):
#     documents: List[str] = Field(..., description="List of document URLs to ingest and embed.")

# class IngestResponse(BaseModel):
#     message: str
#     documents_ingested: int

# # --- LLM System Prompt ---
# SYSTEM_PROMPT = """
# You are an expert policy analyst. Your task is to answer questions based SOLELY on the provided policy document context.
# Do not use any outside knowledge. If the answer is not available in the provided context, state that you cannot find the answer.
# Provide concise and direct answers for each question. Your answer must be specific and include all key details such as numbers 
# (e.g., number of beds, days, months), percentages, and precise conditions mentioned in the text, without adding unnecessary conversational text.

# Context:
# {context}
# """

# # --- RAG System Setup Function ---
# async def setup_rag_system():
#     """Initializes all RAG components, now with selectable LLM providers."""
#     global vector_store, rag_chain

#     print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
#     print(f"Initializing ChromaDB from directory: {PERSIST_DIRECTORY}...")
#     vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
#     llm = None
#     if LLM_PROVIDER == "google":
#         if not GOOGLE_API_KEY:
#             raise ValueError("GOOGLE_API_KEY must be set for the 'google' provider.")
#         print(f"Initializing Google Gemini LLM client: {GOOGLE_MODEL_NAME}...")
#         llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    
#     elif LLM_PROVIDER == "openai":
#         if not OPENAI_API_KEY:
#             raise ValueError("OPENAI_API_KEY must be set for the 'openai' provider.")
#         print(f"Initializing OpenAI LLM client: {OPENAI_MODEL_NAME}...")
#         llm = ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.0)
        
#     # --- ADDED: Support for NVIDIA provider ---
#     elif LLM_PROVIDER == "nvidia":
#         if not NVIDIA_API_KEY:
#             raise ValueError("NVIDIA_API_KEY must be set for the 'nvidia' provider.")
#         print(f"Initializing NVIDIA LLM client: {NVIDIA_MODEL_NAME}...")
#         llm = ChatNVIDIA(model=NVIDIA_MODEL_NAME, nvidia_api_key=NVIDIA_API_KEY, temperature=0.1)
        
#     else:
#         raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'google', 'openai', or 'nvidia'.")

#     prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
#     retriever = vector_store.as_retriever(search_kwargs={"k": 10})
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, document_chain)
    
#     print(f"RAG system setup complete with {LLM_PROVIDER.upper()} LLM.")

# # --- CHANGED: Rewritten Ingestion Logic for PDFs ---
# async def process_ingestion_in_background(document_urls: List[str]):
#     """
#     Downloads documents from URLs, parses PDFs, chunks the content, 
#     and ingests them into ChromaDB.
#     """
#     global vector_store
#     print(f"Starting background ingestion for {len(document_urls)} documents...")
#     ingested_count = 0
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     for url in document_urls:
#         try:
#             print(f"Downloading and processing PDF from: {url}")
#             response = requests.get(url)
#             response.raise_for_status()

#             pdf_file = io.BytesIO(response.content)
#             reader = pypdf.PdfReader(pdf_file)
#             full_text = "".join(page.extract_text() or "" for page in reader.pages)

#             if not full_text:
#                 print(f"Could not extract text from {url}. Skipping.")
#                 continue

#             docs = [Document(page_content=full_text, metadata={"source": url})]
#             chunks = text_splitter.split_documents(docs)
            
#             if vector_store:
#                 vector_store.add_documents(chunks)
#                 # vector_store.persist()
#                 ingested_count += 1
#                 print(f"Successfully processed and ingested chunks from {url}. Chunks added: {len(chunks)}")
#             else:
#                 print("Vector store not initialized during background ingestion.")
#         except Exception as e:
#             print(f"Error during background ingestion of {url}: {e}")
    
#     print(f"Background ingestion finished. Successfully ingested {ingested_count} documents.")
#     # REMOVED: Re-initialization of the RAG system is not needed here. 
#     # The vector_store object updates in place.

# # --- Application Startup Event ---
# @app.on_event("startup")
# async def on_startup():
#     """Initializes the RAG system when the FastAPI application starts."""
#     await setup_rag_system()

# # --- API Endpoints ---

# @app.post("/api/v1/hackrx/run", response_model=QueryResponse, summary="Query policy documents")
# async def run_query(request: QueryRequest, token: str = Depends(verify_token)): # ADDED: Security dependency
#     """
#     Processes a list of questions against the ingested policy documents
#     and returns a list of answers in parallel.
#     """
#     if not rag_chain:
#         raise HTTPException(status_code=503, detail="RAG system not initialized.")
#     if not request.questions:
#         raise HTTPException(status_code=400, detail="No questions provided.")

#     tasks = [rag_chain.ainvoke({"input": q}) for q in request.questions]
#     answers_list: List[Answer] = []
#     try:
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         for result in results:
#             if isinstance(result, Exception):
#                 answers_list.append(Answer(answer=f"Error processing question: {result}"))
#             else:
#                 answers_list.append(Answer(answer=result.get("answer", "Could not generate an answer.")))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error during query processing: {e}")

#     return QueryResponse(answers=answers_list)

# @app.post("/api/v1/hackrx/ingest", response_model=IngestResponse, summary="Ingest documents into the system")
# async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_token)): # ADDED: Security dependency
#     """
#     Ingests documents from URLs into the vector store as a background task.
#     """
#     if not request.documents:
#         raise HTTPException(status_code=400, detail="No document URLs provided.")

#     background_tasks.add_task(process_ingestion_in_background, request.documents)

#     return IngestResponse(
#         message="Document ingestion started in the background.",
#         documents_ingested=len(request.documents)
#     )

# # uvicorn main:app --host 0.0.0.0 --port 8000 --reload



import os
import io
import requests
import pypdf
import asyncio
import warnings
warnings.simplefilter(action='ignore')
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA # Only NVIDIA imports
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document


# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
# Using NVIDIA's embedding model
EMBEDDING_MODEL_NAME = "nvidia/embed-qa-4" 
LLM_PROVIDER = "nvidia" # Fixed to 'nvidia' as only NVIDIA is configured
# NVIDIA LLM Model
NVIDIA_MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1" 

# --- Load environment variables ---
load_dotenv()
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="API for processing policy documents and answering questions using RAG.",
    version="1.0.0",
)

# --- Security Configuration ---
auth_scheme = HTTPBearer()
EXPECTED_TOKEN = "5639d6715d8cfe974e313a8fe74f2394761238a54ba9d15d550d11a8e5a767ee"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Dependency function to verify the Bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid authorization token",
        )
    return credentials.credentials

# --- Global variables for the RAG components ---
vector_store = None
rag_chain = None
llm_model = None # Renamed from 'model' to avoid conflict with pydantic 'model'

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document URLs to query (currently supports one).")
    questions: List[str] = Field(..., description="List of natural language questions to answer.")

class Answer(BaseModel):
    answer: str

class QueryResponse(BaseModel):
    answers: List[Answer]

class IngestRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document URLs to ingest and embed.")

class IngestResponse(BaseModel):
    message: str
    documents_ingested: int

# --- LLM System Prompt ---
SYSTEM_PROMPT = """
You are an expert policy analyst. Your task is to answer questions based SOLELY on the provided policy document context.
Do not use any outside knowledge. If the answer is not available in the provided context, state that you cannot find the answer.
Provide concise and direct answers for each question. Your answer must be specific and include all key details such as numbers 
(e.g., number of beds, days, months), percentages, and precise conditions mentioned in the text, without adding unnecessary conversational text.

Context:
{context}
"""

# --- Text Splitter Configuration ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def load_document_from_url(url: str) -> List[Document]:
    """
    Downloads a PDF from a URL, extracts its text, and returns it as a list of LangChain Document objects.
    Args:
        url (str): The URL of the PDF document.
    Returns:
        List[Document]: A list of LangChain Document objects from the document.
    """
    print(f"Attempting to download document from: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        print("Document downloaded successfully. Processing PDF...")
        pdf_file = io.BytesIO(response.content)
        
        reader = pypdf.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        
        print(f"Extracted {len(full_text)} characters from the document.")

        # Create a single Document object for the full text
        doc = Document(page_content=full_text, metadata={"source": url})
        
        # Split the document into smaller chunks
        chunks = text_splitter.split_documents([doc]) # Use split_documents for Document objects
        print(f"Document split into {len(chunks)} chunks.")
        
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download document from URL. {e}")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred during document processing. {e}")
        return []

# --- RAG System Setup Function ---
async def setup_rag_system():
    """Initializes all RAG components."""
    global vector_store, rag_chain, llm_model

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} (NVIDIAEmbeddings)...")
    # Use NVIDIAEmbeddings directly
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY must be set for NVIDIAEmbeddings.")
    embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL_NAME, nvidia_api_key=NVIDIA_API_KEY)

    print(f"Initializing ChromaDB from directory: {PERSIST_DIRECTORY}...")
    # Initialize ChromaDB. It will load existing data or create a new one.
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # Initialize NVIDIA LLM client
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY must be set for the 'nvidia' provider.")
    print(f"Initializing NVIDIA LLM client: {NVIDIA_MODEL_NAME}...")
    llm_model = ChatNVIDIA(model=NVIDIA_MODEL_NAME, nvidia_api_key=NVIDIA_API_KEY, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    document_chain = create_stuff_documents_chain(llm_model, prompt) # Use llm_model
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"RAG system setup complete with NVIDIA LLM and NVIDIA Embeddings.")

# --- Ingestion Logic ---
async def process_ingestion_in_background(document_urls: List[str]):
    """
    Downloads documents from URLs, parses PDFs, chunks the content, 
    and ingests them into ChromaDB.
    """
    global vector_store
    print(f"Starting background ingestion for {len(document_urls)} documents...")
    ingested_count = 0

    for url in document_urls:
        try:
            print(f"Downloading and processing PDF from: {url}")
            chunks = load_document_from_url(url) 
            
            if not chunks:
                print(f"No chunks extracted from {url}. Skipping.")
                continue

            if vector_store:
                vector_store.add_documents(chunks)
                # ChromaDB's persist() needs to be called to write changes to disk
                vector_store.persist() 
                ingested_count += 1
                print(f"Successfully processed and ingested chunks from {url}. Chunks added: {len(chunks)}")
            else:
                print("Vector store not initialized during background ingestion.")
        except Exception as e:
            print(f"Error during background ingestion of {url}: {e}")
    
    print(f"Background ingestion finished. Successfully ingested {ingested_count} documents.")

# --- Application Startup Event ---
@app.on_event("startup")
async def on_startup():
    """Initializes the RAG system when the FastAPI application starts."""
    await setup_rag_system()

# --- API Endpoints ---

@app.get("/", summary="Health check endpoint")
async def root():
    """
    Returns a simple message to indicate that the server is running.
    """
    return {"message": "Server is running!"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, summary="Query policy documents")
async def run_query(request: QueryRequest, token: str = Depends(verify_token)): # Security dependency
    """
    Processes a list of questions against the ingested policy documents
    and returns a list of answers in parallel.
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please try again in a moment.")
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided.")

    tasks = [rag_chain.ainvoke({"input": q}) for q in request.questions]
    answers_list: List[Answer] = []
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                answers_list.append(Answer(answer=f"Error processing question: {result}"))
            else:
                # Assuming result is a dict from create_retrieval_chain, containing 'answer' key
                generated_answer = result.get("answer", "Could not generate an.answer.")
                answers_list.append(Answer(answer=generated_answer))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during query processing: {e}")

    return QueryResponse(answers=answers_list)