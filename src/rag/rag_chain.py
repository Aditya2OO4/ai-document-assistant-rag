from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI


class RAGChain:
    def __init__(self):
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.2
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.page_content for d in docs)
        prompt = self.prompt.format(context=context, question=question)

        response = self.llm.invoke(prompt)
        return response.content
