import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaVectorStore:
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        # Get API Key from Render Environment Variables
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not self.api_key:
            print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not set!")

        # Using the API endpoint saves 500MB+ of RAM on Render
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

    def add_documents(self, documents):
        vector_store = self.get_vector_store()
        vector_store.add_documents(documents)
        # Chroma 0.4+ persists automatically, but call is kept for compatibility
        vector_store.persist()
