from fastapi import FastAPI, Body
import os
from google import genai
from langchain_postgres import PGVector

app = FastAPI()

# Env vars
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def embed_text(text: str):
    emb = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return emb.embedding

# LangChain embedding wrapper
class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]
    def embed_query(self, text):
        return embed_text(text)

embeddings = GeminiEmbeddings()

# Neon vector DB
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
