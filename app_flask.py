from flask import Flask, render_template, request
import os

from src.ingestion.document_loader import PDFDocumentLoader
from src.ingestion.text_splitter import TextSplitter
from src.vectorstore.chroma_manager import ChromaVectorStore
from src.retrieval.retriever import VectorRetriever
from src.rag.rag_chain import RAGChain

app = Flask(__name__)
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        pdf = request.files["pdf"]
        question = request.form["question"]

        pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        pdf.save(pdf_path)

        loader = PDFDocumentLoader(pdf_path)
        docs = loader.load()

        splitter = TextSplitter()
        chunks = splitter.split(docs)

        vectorstore = ChromaVectorStore()
        vectorstore.create_store(chunks)

        retriever = VectorRetriever(vectorstore.load_store())
        retrieved_docs = retriever.retrieve(question)

        rag = RAGChain()
        answer = rag.generate_answer(question, retrieved_docs)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


