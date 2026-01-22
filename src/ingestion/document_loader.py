from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_core.documents import Document
import os


class PDFDocumentLoader:

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if not self.file_path.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")

    def load(self):
        # First attempt: PyPDF (fast, but fragile)
        try:
            loader = PyPDFLoader(self.file_path)
            return loader.load()

        except Exception as e:
            print("⚠️ PyPDF failed, falling back to PDFMiner:", e)

        # Fallback: PDFMiner (slower, but robust)
        try:
            loader = PDFMinerLoader(self.file_path)
            return loader.load()

        except Exception as e:
            raise ValueError(
                f"Failed to load PDF from {self.file_path}. "
                f"Likely a malformed or web-generated PDF."
            ) from e
