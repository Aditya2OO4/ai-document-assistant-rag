import os
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class RAGChain:
    def __init__(self, vector_store):
        # 1. Setup the LLM using the more stable partner package
        self.llm = ChatMistralAI(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-latest",
            temperature=0.2
        )

        # 2. Define your CUSTOM PROMPT here (This replaces your ChatMessage logic)
        self.system_prompt = (
            "You are a helpful assistant. "
            "Answer ONLY using the provided context."
            "\n\n"
            "{context}"
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

        # 3. Create the Chain
        # This handles the joining of 'docs' into 'context' automatically
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        self.rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    def ask(self, query: str):
        # This returns a dict with 'answer' and 'context'
        response = self.rag_chain.invoke({"input": query})
        return response["answer"]
