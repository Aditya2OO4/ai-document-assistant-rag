from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from typing import List
import os


class ChromaVectorStore:
    """
    Handles embedding generation and vector storage using ChromaDB.
    """

    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        os.makedirs(self.persist_directory, exist_ok=True)

    def create_store(self, documents: List[Document]) -> Chroma:
        """
        Creates and persists a vector store from documents.
        """
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )

        vectordb.persist()
        return vectordb

    def load_store(self) -> Chroma:
        """
        Loads an existing vector store from disk.
        """
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
