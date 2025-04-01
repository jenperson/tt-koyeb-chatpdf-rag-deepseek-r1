from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings  # Keeping Ollama Embeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")

class DeepSeekLLM:
    """
    Use the DeepSeek model served from Koyeb
    """
    def __init__(self, api_url: str = f"{DEEPSEEK_API_URL}", model: str = "DeepSeek-R1-Distill-Llama-8B"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self.model = model

    def invoke(self, formatted_input: str, temperature: float = 1.0,  max_tokens: int = 500, role: str = "user"):
        """Send a request to the OpenAI API and return the response."""
        if not isinstance(formatted_input, str):
            formatted_input = formatted_input.to_string()

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": role, "content": formatted_input}],
                model=f"/models/{self.model}",
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = DeepSeekLLM()        
        self.embeddings = OllamaEmbeddings(model=embedding_model)  # Keeping Ollama embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2, temperature: float = 1.0, max_tokens: int = 500):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model.invoke      # Queries the DeepSeek API
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the DeepSeek LLM.")
        return chain.invoke(formatted_input, temperature=temperature, max_tokens=max_tokens)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None