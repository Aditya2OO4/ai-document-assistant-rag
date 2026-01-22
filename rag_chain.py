import os
import google.generativeai as genai
from typing import List
from langchain_core.documents import Document


class RAGChain:
    """
    Combines retrieved documents with Gemini LLM
    to generate grounded answers.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _build_context(self, documents: List[Document]) -> str:
        """
        Combines retrieved document chunks into a single context string.
        """
        context = "\n\n".join(
            f"Source {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        )
        return context

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generates an answer using retrieved context + user query.
        """
        context = self._build_context(documents)

        prompt = f"""
You are an AI assistant. Answer the question strictly
using the context provided below.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.model.generate_content(prompt)
        return response.text.strip()
