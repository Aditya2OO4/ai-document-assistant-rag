import os
import tempfile
from flask import Flask, render_template, request
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore.chroma_manager import ChromaVectorStore
from src.rag.rag_chain import RAGChain

app = Flask(__name__)

# Disable ChromaDB telemetry to stop those "capture" errors in logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    
    if request.method == "POST":
        # 1. Handle File Upload
        if 'file' not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", error="No selected file")

        # 2. Process PDF
        if file:
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    file.save(temp_file.name)
                    temp_path = temp_file.name

                # Load and Split PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(docs)

                # 3. Initialize Vector Store (With Error Handling)
                try:
                    vectorstore_manager = ChromaVectorStore()
                    vectorstore = vectorstore_manager.create_store(chunks)
                except Exception as e:
                    print(f"Vector Store Error: {e}")
                    return render_template("index.html", error=f"Database Error: {str(e)}")

                # 4. Generate Answer
                query = request.form.get("query", "Summarize this document.")
                rag_chain = RAGChain(vectorstore)
                answer = rag_chain.ask(query)

            except Exception as e:
                print(f"Processing Error: {e}")
                error = f"An error occurred: {str(e)}"
            
            finally:
                # Cleanup: Delete the temporary file
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)

    return render_template("index.html", answer=answer, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
