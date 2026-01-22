from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class TextSplitter:
    """
    Splits documents into semantically meaningful chunks
    suitable for embedding and vector storage.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into chunks while preserving metadata.
        """
        return self.splitter.split_documents(documents)
