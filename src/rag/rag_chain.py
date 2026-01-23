from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI


class RAGChain:
    """
    Performs Retrieval Augmented Generation using Mistral.
    """

    def __init__(self):
        # Initialize Mistral chat model
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.2
        )

        # Simple template
        self.template = 
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.template
        )

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        context_text = "\n\n".join([d.page_content for d in docs])
        prompt_text = self.prompt.format(context=context_text, question=question)

        # Ask Mistral
        answer = self.llm.chat([{"role": "user", "content": prompt_text}])
        return answer.content
