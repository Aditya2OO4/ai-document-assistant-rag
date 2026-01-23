import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaVectorStore:
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        # Initialize the API-based embedding model
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

    # THIS IS THE MISSING METHOD CAUSING THE ERROR
    def create_store(self, documents):
        """Creates a new vector store from a list of documents."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        # In newer versions persist is automatic, but we call it for safety
        self.vector_store.persist()
        return self.vector_store

    def add_documents(self, documents):
        vector_store = self.get_vector_store()
        vector_store.add_documents(documents)
        vector_store.persist()
