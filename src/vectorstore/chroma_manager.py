import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaVectorStore:
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not self.api_key:
            print("CRITICAL: HUGGINGFACEHUB_API_TOKEN is missing!")

        # Using the Endpoint class fixes the .post() AttributeError
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=self.api_key,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        return self.vector_store

    def create_store(self, documents):
        """Creates a new vector store from documents."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        # Chroma handles persistence automatically in recent versions,
        # but this ensures the 'db' folder is written.
        return self.vector_store
