from typing import List
from langchain.schema import Document
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os


class RAGChain:
    def __init__(self):
        self.client = MistralClient(
            api_key=os.environ["MISTRAL_API_KEY"]
        )

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.page_content for d in docs)

        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. Answer ONLY using the provided context."
            ),
            ChatMessage(
                role="user",
                content=f"""
Context:
{context}

Question:
{question}

Answer:
"""
            )
        ]

        response = self.client.chat(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content
