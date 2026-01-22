from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class VectorRetriever:
    """
    Handles similarity search over the vector database.
    """

    def __init__(self, vectorstore: Chroma, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """
        Returns top-k most similar document chunks for a query.
        """
        return self.vectorstore.similarity_search(query, k=self.top_k)
