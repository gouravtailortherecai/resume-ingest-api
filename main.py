# ingest_api/main.py
from fastapi import FastAPI, Body
import os
from langchain_groq import GroqEmbeddings
from langchain_postgres import PGVector

app = FastAPI()

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")

# Groq Embeddings
embeddings = GroqEmbeddings(
    model="nomic-embed-text",
    groq_api_key=GROQ_API_KEY
)

# Neon vector DB
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)

@app.post("/ingest")
def ingest_resume(resume_text: str = Body(..., embed=True)):
    """
    Take resume text, embed with Groq, and store in Neon.
    """
    try:
        docs = [{"page_content": resume_text, "metadata": {"source": "resume"}}]
        vectorstore.add_documents(docs)
        return {"status": "success", "message": "Resume stored in Neon"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
