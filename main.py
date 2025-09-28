# main.py (Ingest API)
from fastapi import FastAPI, Body
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import os

app = FastAPI()

# Environment variable
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)

@app.post("/ingest")
def ingest_resume(resume_text: str = Body(..., embed=True)):
    try:
        docs = [{"page_content": resume_text, "metadata": {"source": "resume"}}]
        vectorstore.add_documents(docs)
        return {"status": "success", "message": "Resume stored in Neon"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
