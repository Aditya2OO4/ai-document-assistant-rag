import os
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaManager:
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        # Use Inference API to save memory on Render
        # Ensure HUGGINGFACEHUB_API_TOKEN is set in Render Environment Variables
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not self.api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=self.api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
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
        vector_store.persist()
