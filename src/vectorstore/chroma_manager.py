from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
import os


class ChromaVectorStore:
    """
    Handles embedding generation and vector storage using Gemini embeddings.
    Safe for low-memory deployments.
    """

    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory

        # ✅ API-based embeddings (NO torch, NO CUDA)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        os.makedirs(self.persist_directory, exist_ok=True)

    def create_store(self, documents: List[Document]) -> Chroma:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        return vectordb

    def load_store(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
