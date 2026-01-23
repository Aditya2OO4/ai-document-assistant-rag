import os
import logging
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Configure logging to see exactly what happens on Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.embedding_model = None
        
        # 1. Try Hugging Face API (Best for Render 512MB RAM)
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if api_key:
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                logger.info("Attempting to use HuggingFace Inference API...")
                self.embedding_model = HuggingFaceEndpointEmbeddings(
                    huggingfacehub_api_token=api_key,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    task="feature-extraction"
                    # endpoint_url removed to fix Pydantic error
                )
                logger.info("Successfully initialized HuggingFace API Embeddings.")
            except Exception as e:
                logger.error(f"Failed to load HF API Embeddings: {e}")
                self.embedding_model = None

        # 2. Fallback: Local SentenceTransformers (Uses CPU/RAM)
        if self.embedding_model is None:
            logger.warning("Falling back to local SentenceTransformers. WARNING: High RAM usage.")
            try:
                # We use the standard HuggingFaceEmbeddings which runs locally
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Successfully initialized Local Embeddings.")
            except Exception as e:
                logger.critical(f"CRITICAL: Could not load ANY embedding model: {e}")
                raise e

    def get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        return self.vector_store

    def create_store(self, documents):
        """Creates a new vector store from documents."""
        logger.info(f"Creating vector store with {len(documents)} documents...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        return self.vector_store

    def add_documents(self, documents):
        vector_store = self.get_vector_store()
        vector_store.add_documents(documents)
