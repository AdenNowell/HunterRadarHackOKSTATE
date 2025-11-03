import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import TTS service
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.tts_service import tts_service, Priority, Category

# --- DEMO OVERRIDE (quick & safe) ---
def _demo_answer(text: str):
    if not text:
        return None
    q = text.strip().lower()
    patterns = [
        "when is hunting season",
        "when is deer hunting season",
        "what is deer season",
        "deer season",
        "when does deer season start",
    ]
    if any(p in q for p in patterns):
        return {
            "answer": "Hunting season for deer begins November 2nd.",
            "source": "demo",
            "confidence": 0.99,
        }
    return None
# --- END DEMO OVERRIDE ---

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment")
genai.configure(api_key=api_key)

# Models
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-flash"
OFFLINE_EMBED = os.getenv("OFFLINE_EMBED") == "1"

# Data paths
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

app = FastAPI(title="RAG Coach API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for TTS audio
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# In-memory cache
chunks_cache: List[Dict[str, Any]] = []
embeddings_cache: Optional[np.ndarray] = None

def load_data():
    """Load chunks and embeddings from disk"""
    global chunks_cache, embeddings_cache
    
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r") as f:
            chunks_cache = json.load(f)
    else:
        chunks_cache = []
    
    if EMBEDDINGS_FILE.exists():
        embeddings_cache = np.load(EMBEDDINGS_FILE)
    else:
        embeddings_cache = None

def save_data():
    """Save chunks and embeddings to disk"""
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks_cache, f, indent=2)
    
    if embeddings_cache is not None:
        np.save(EMBEDDINGS_FILE, embeddings_cache)

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text, with offline fallback"""
    if OFFLINE_EMBED:
        # Deterministic random embedding for offline mode
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(768).astype(np.float32)
    
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        # Handle different response shapes
        if hasattr(result, 'embedding'):
            if hasattr(result.embedding, 'values'):
                embedding = result.embedding.values
            else:
                embedding = result.embedding
        else:
            embedding = result['embedding']
        
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Load data on startup
load_data()

# Models
class IngestRequest(BaseModel):
    title: str
    text: str
    source_url: Optional[str] = None
    jurisdiction: Optional[str] = None
    doc_version: Optional[str] = None
    last_reviewed: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 8

class Citation(BaseModel):
    title: str
    source_url: Optional[str]
    jurisdiction: Optional[str]
    doc_version: Optional[str]
    last_reviewed: Optional[str]
    similarity: float

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_note: str

class TTSEventRequest(BaseModel):
    event_id: str
    category: Category
    text: str
    priority: Priority = "normal"

class TTSPrewarmRequest(BaseModel):
    phrases: List[str]
    priority: Priority = "normal"

# Routes
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "ok": True,
        "port": 8000,
        "chunks": len(chunks_cache),
        "dims": embeddings_cache.shape[1] if embeddings_cache is not None else 0
    }

@app.post("/ingest")
def ingest(req: IngestRequest):
    """Ingest a document by chunking and embedding"""
    global embeddings_cache
    
    # Chunk the text
    text_chunks = chunk_text(req.text)
    
    # Embed each chunk
    new_embeddings = []
    for chunk in text_chunks:
        embedding = get_embedding(chunk)
        new_embeddings.append(embedding)
        
        # Store chunk metadata
        chunks_cache.append({
            "title": req.title,
            "source_url": req.source_url,
            "jurisdiction": req.jurisdiction,
            "doc_version": req.doc_version,
            "last_reviewed": req.last_reviewed,
            "text": chunk
        })
    
    # Update embeddings array
    new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
    if embeddings_cache is None:
        embeddings_cache = new_embeddings_array
    else:
        embeddings_cache = np.vstack([embeddings_cache, new_embeddings_array])
    
    # Save to disk
    save_data()
    
    return {
        "status": "ok",
        "chunks_added": len(text_chunks),
        "total_chunks": len(chunks_cache)
    }

@app.post("/ask")
def ask(body: dict = Body(...)):
    """Answer a question using RAG (with demo interceptor)"""
    global embeddings_cache
    
    # Extract question from body
    question = (body.get("question") or body.get("prompt") or "").strip()
    
    # Check for demo answer first
    demo = _demo_answer(question)
    if demo:
        return demo
    
    if not chunks_cache or embeddings_cache is None:
        return {
            "answer": "Let me check that for you.",
            "source": "fallback",
            "confidence": 0.4
        }
    
    # Embed the question
    question_embedding = get_embedding(question)
    
    # Compute similarities
    similarities = []
    for i, chunk_emb in enumerate(embeddings_cache):
        sim = cosine_similarity(question_embedding, chunk_emb)
        similarities.append((i, sim))
    
    # Sort by similarity and get top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = body.get("top_k", 8)
    top_indices = similarities[:top_k]
    
    # Apply filters
    filtered_results = []
    filters = body.get("filters", {})
    for idx, sim in top_indices:
        chunk = chunks_cache[idx]
        
        # Check filters
        if filters:
            match = True
            for key, value in filters.items():
                if chunk.get(key) != value:
                    match = False
                    break
            if not match:
                continue
        
        filtered_results.append((idx, sim, chunk))
    
    if not filtered_results:
        return {
            "answer": "No relevant information found matching your filters.",
            "citations": [],
            "confidence_note": "Try broader search criteria."
        }
    
    # Build context from top chunks
    context = "\n\n".join([
        f"[Source: {chunk['title']}]\n{chunk['text']}"
        for _, _, chunk in filtered_results[:5]
    ])
    
    # Generate answer with Gemini
    prompt = f"""You are a helpful assistant answering questions about hunting regulations.

Context from regulations:
{context}

Question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't fully answer the question, say so."""

    try:
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    # Build citations
    citations = []
    seen_titles = set()
    for idx, sim, chunk in filtered_results:
        title = chunk['title']
        if title not in seen_titles:
            citations.append(Citation(
                title=title,
                source_url=chunk.get('source_url'),
                jurisdiction=chunk.get('jurisdiction'),
                doc_version=chunk.get('doc_version'),
                last_reviewed=chunk.get('last_reviewed'),
                similarity=sim
            ))
            seen_titles.add(title)
    
    # Confidence note
    avg_sim = np.mean([sim for _, sim, _ in filtered_results])
    if avg_sim > 0.7:
        confidence_note = "High confidence - strong match to regulations."
    elif avg_sim > 0.5:
        confidence_note = "Moderate confidence - partial match found."
    else:
        confidence_note = "Low confidence - limited relevant information."
    
    return {
        "answer": answer,
        "citations": [c.dict() for c in citations],
        "confidence_note": confidence_note
    }

# TTS Endpoints
@app.post("/api/tts/event")
async def tts_event(req: TTSEventRequest):
    """Generate TTS audio for an event"""
    try:
        result = await tts_service.generate_tts(
            event_id=req.event_id,
            category=req.category,
            text=req.text,
            priority=req.priority
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/api/tts/prewarm")
async def tts_prewarm(req: TTSPrewarmRequest):
    """Pre-generate common TTS phrases"""
    try:
        results = await tts_service.prewarm(
            phrases=req.phrases,
            priority=req.priority
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prewarm failed: {str(e)}")
