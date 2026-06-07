"""
agent/rag_pipeline.py

Production RAG pipeline built entirely on LangChain abstractions.

┌─────────────────────────────────────────────────────────────────────┐
│  Vector Store Backends (LangChain)                                   │
│  ─────────────────────────────────                                   │
│  faiss   → langchain_community.vectorstores.FAISS                   │
│  chroma  → langchain_chroma.Chroma                                  │
│                                                                      │
│  Embedding Classes (LangChain)                                       │
│  ─────────────────────────────                                       │
│  tfidf   → TFIDFEmbeddings  (custom, langchain_core.Embeddings)     │
│  hf      → HuggingFaceEmbeddings (langchain_huggingface)            │
│  openai  → OpenAIEmbeddings  (langchain_openai)                     │
│  fake    → DeterministicFakeEmbedding (tests / offline)             │
└─────────────────────────────────────────────────────────────────────┘

Environment variables:
  RAG_BACKEND    = faiss | chroma           (default: faiss)
  EMBED_BACKEND  = tfidf | hf | openai | fake  (default: tfidf)
  RAG_TOP_K      = int                      (default: 3)
  EMBED_MODEL    = HF model name            (default: all-MiniLM-L6-v2)

Persist paths:
  ./vector_store/faiss_lc/     ← LangChain FAISS index folder
  ./vector_store/chroma_lc/    ← LangChain Chroma persist directory
"""

from __future__ import annotations

import json
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ── LangChain vector stores ───────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS          # pip: langchain-community + faiss-cpu
from langchain_chroma import Chroma                         # pip: langchain-chroma

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent
KB_PATH     = BASE_DIR / "knowledge_base" / "autostream_kb.json"
VECTOR_DIR  = BASE_DIR / "vector_store"
FAISS_DIR   = VECTOR_DIR / "faiss_lc"
CHROMA_DIR  = VECTOR_DIR / "chroma_lc"

VECTOR_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

#RAG_BACKEND   = os.getenv("RAG_BACKEND",   "faiss").lower()   # faiss | chroma
#EMBED_BACKEND = os.getenv("EMBED_BACKEND", "tfidf").lower()   # tfidf | hf | openai | fake
#TOP_K         = int(os.getenv("RAG_TOP_K", "3"))
#EMBED_MODEL   = os.getenv("EMBED_MODEL",   "all-MiniLM-L6-v2")

# ── Singletons (cached per process) ──────────────────────────────────────────

_faiss_store:  Optional[FAISS]  = None
_chroma_store: Optional[Chroma] = None
_embeddings:   Optional[Embeddings] = None


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — KNOWLEDGE BASE  →  LangChain Documents
# ══════════════════════════════════════════════════════════════════════════════

def _load_kb() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents() -> List[Document]:
    """
    Converts the AutoStream KB JSON into a list of LangChain Document objects.
    Each Document has:
        page_content : rich text chunk (better semantic embeddings)
        metadata     : {id, category, ...} for filtering
    """
    kb     = _load_kb()
    docs: List[Document] = []

    # ── Company overview ──────────────────────────────────────────────────────
    c = kb["company"]
    docs.append(Document(
        page_content=(
            f"AutoStream Company Overview\n"
            f"Name: {c['name']}\n"
            f"Tagline: {c['tagline']}\n"
            f"Description: {c['description']}"
        ),
        metadata={"id": "company_overview", "category": "company"},
    ))

    # ── Pricing plans ─────────────────────────────────────────────────────────
    for plan in kb["pricing"]["plans"]:
        feats = "\n  • " + "\n  • ".join(plan["features"])
        lims  = ""
        if plan["limitations"]:
            lims = "\nLimitations:\n  • " + "\n  • ".join(plan["limitations"])
        docs.append(Document(
            page_content=(
                f"{plan['name']} — Pricing and Features\n"
                f"Price: ${plan['price_monthly']} per month\n"
                f"Features:{feats}{lims}"
            ),
            metadata={
                "id":       f"pricing_{plan['name'].lower().replace(' ','_')}",
                "category": "pricing",
                "plan":     plan["name"],
            },
        ))

    # ── Policies ──────────────────────────────────────────────────────────────
    label_map = {
        "refund_policy":       "Refund Policy",
        "support_policy":      "Support Policy",
        "cancellation_policy": "Cancellation Policy",
        "data_policy":         "Data Policy",
    }
    for key, label in label_map.items():
        if key in kb.get("policies", {}):
            docs.append(Document(
                page_content=f"AutoStream {label}\n{kb['policies'][key]}",
                metadata={"id": key, "category": "policy", "policy_type": key},
            ))

    # ── FAQ ───────────────────────────────────────────────────────────────────
    for i, faq in enumerate(kb.get("faq", [])):
        docs.append(Document(
            page_content=(
                f"FAQ\nQuestion: {faq['question']}\nAnswer: {faq['answer']}"
            ),
            metadata={"id": f"faq_{i}", "category": "faq"},
        ))

    return docs


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — EMBEDDING CLASS  (LangChain Embeddings interface)
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFEmbeddings(Embeddings):
    """
    LangChain-compatible TF-IDF embedding class (fully offline).

    Implements the Embeddings interface:
        embed_documents(texts) → List[List[float]]
        embed_query(text)      → List[float]

    The vectorizer is fit lazily on first call to embed_documents()
    and then reused for embed_query() so both share the same vocabulary.
    """

    def __init__(self, ngram_range=(1, 2), max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self._fitted = False

    def _l2_normalize(self, matrix) -> np.ndarray:
        mat   = np.array(matrix, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        # This is a NumPy function that calculates the vector norm (magnitude). By default, it calculates the L2 norm, which is the square root of the sum of the squares of the vector's components
        # This is a critical parameter for the next line of code. If your input matrix mat has a shape of (number_of_docs, dimensions), using keepdims=True ensures the output norms has a shape of (number_of_docs, 1) instead of a flat array (number_of_docs,).
        norms[norms == 0] = 1.0#This is excellent practice. It handles the "zero-vector" edge case (e.g., if a document contains only stop-words or empty text) to prevent a DivisionByZero error.
        return mat / norms
    def fit_on_corpus(self, texts: List[str]):
        """Explicitly fit on full corpus (critical fix)."""
        if not self._fitted:
            self._vectorizer.fit(texts)
            self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Fit (first call) or transform (subsequent calls) the TF-IDF vectorizer."""
        if not self._fitted:
            mat = self._vectorizer.fit_transform(texts).toarray()
            self._fitted = True# This is excellent practice. It handles the "zero-vector" edge case (e.g., if a document contains only stop-words or empty text) to prevent a DivisionByZero error.
        else:
            mat = self._vectorizer.transform(texts).toarray()
        return self._l2_normalize(mat).tolist()

    def embed_query(self, text: str) -> List[float]:
         if not self._fitted:
            # Safety: fit on full corpus before any query
            docs = build_documents()
            corpus_texts = [doc.page_content for doc in docs]
            self.fit_on_corpus(corpus_texts)
         mat = self._vectorizer.transform([text]).toarray()
         return self._l2_normalize(mat)[0].tolist()

def get_embeddings() -> Embeddings:
    """
    Returns the configured LangChain Embeddings instance (cached singleton).

    EMBED_BACKEND options:
      tfidf  → TFIDFEmbeddings (offline, no API key, scikit-learn)
      hf     → HuggingFaceEmbeddings (sentence-transformers, needs internet)
      openai → OpenAIEmbeddings (needs OPENAI_API_KEY)
      fake   → DeterministicFakeEmbedding (unit tests)
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    #backend = EMBED_BACKEND

    #if backend == "hf":
    from langchain_huggingface import HuggingFaceEmbeddings
    #print(f"[RAG] Using HuggingFaceEmbeddings: all-MiniLM-L6-v2")
    #_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #elif backend == "openai":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    #api_key=st.secrets["api_key1"]
    #model = EMBED_MODEL if "text-embedding" in EMBED_MODEL else "text-embedding-3-small"
    #print(f"[RAG] Using OpenAIEmbeddings: {gemini-embedding}")
    #_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",api_key=api_key)

    #elif backend == "fake":
    #from langchain_community.embeddings import DeterministicFakeEmbedding
    #print("[RAG] Using DeterministicFakeEmbedding (test mode)")
    #_embeddings = DeterministicFakeEmbedding(size=256)

    #else:  # tfidf (default)
    print("[RAG] Using TFIDFEmbeddings (local, offline)")
    _embeddings = TFIDFEmbeddings()

    return _embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3A — FAISS  (langchain_community.vectorstores.FAISS)
# ══════════════════════════════════════════════════════════════════════════════

def _get_faiss_store() -> FAISS:
    """
    Returns a LangChain FAISS vector store.
    Loads from disk if the index folder exists, otherwise builds and saves.
    """
    global _faiss_store

    if _faiss_store is not None:
        return _faiss_store

    embeddings = get_embeddings()

    if FAISS_DIR.exists():
        print(f"[RAG][FAISS] Loading from {FAISS_DIR}")
        # FAISS.load_local requires allow_dangerous_deserialization=True
        # because it uses pickle — safe here since we wrote the index ourselves
        _faiss_store = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[RAG][FAISS] Loaded ({_faiss_store.index.ntotal} vectors)")
        return _faiss_store

    # Build from KB documents
    print("[RAG][FAISS] Building index from knowledge base…")
    docs         = build_documents()
    _faiss_store = FAISS.from_documents(docs, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    _faiss_store.save_local(str(FAISS_DIR))
    print(f"[RAG][FAISS] Indexed {len(docs)} documents → {FAISS_DIR}")
    return _faiss_store


def _retrieve_faiss(query: str, top_k: int = 3) -> str:
    store   = _get_faiss_store()
    results = store.similarity_search(query, k=top_k)
    return "\n\n---\n\n".join(r.page_content for r in results) if results else ""


def _rebuild_faiss():
    global _faiss_store, _embeddings
    _faiss_store = None
    _embeddings  = None          # reset so TF-IDF is re-fit on fresh corpus
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    _get_faiss_store()
    print("[RAG][FAISS] Index rebuilt.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3B — CHROMA  (langchain_chroma.Chroma)
# ══════════════════════════════════════════════════════════════════════════════

def _get_chroma_store() -> Chroma:
    """
    Returns a LangChain Chroma vector store.
    Loads from disk if the persist directory exists, otherwise builds and saves.
    """
    global _chroma_store

    if _chroma_store is not None:
        return _chroma_store

    embeddings = get_embeddings()

    if CHROMA_DIR.exists():
        print(f"[RAG][Chroma] Loading from {CHROMA_DIR}")
        store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        count = store._collection.count()
        if count > 0:
            print(f"[RAG][Chroma] Loaded ({count} documents)")
            _chroma_store = store
            return _chroma_store
        # Empty collection — rebuild
        shutil.rmtree(CHROMA_DIR)
    #Resetting a Vector Database: In AI development, ChromaDB saves its embeddings, collection data, and metadata locally on the disk. Developers often use shutil.rmtree(CHROMA_DIR) to completely wipe the database and start with a clean slate during testing, debugging, or re-indexing.
    # Build from KB documents
    print("[RAG][Chroma] Building collection from knowledge base…")
    docs          = build_documents()
    _chroma_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[RAG][Chroma] Indexed {len(docs)} documents → {CHROMA_DIR}")
    return _chroma_store


def _retrieve_chroma(query: str, top_k: int = 3) -> str:
    store   = _get_chroma_store()
    results = store.similarity_search(query, k=top_k)
    return "\n\n---\n\n".join(r.page_content for r in results) if results else ""


def _rebuild_chroma():
    global _chroma_store, _embeddings
    _chroma_store = None
    _embeddings   = None
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    _get_chroma_store()
    print("[RAG][Chroma] Collection rebuilt.")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (imported by agent/graph.py — unchanged interface)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Main retrieval entry point used by the LangGraph agent.

    Dispatches to the backend selected by RAG_BACKEND env var:
      faiss  → LangChain FAISS (langchain_community.vectorstores.FAISS)
      chroma → LangChain Chroma (langchain_chroma.Chroma)

    Both backends share the same LangChain Embeddings instance
    so indexing and query embeddings are always consistent.

    Args:
        query : raw user message
        top_k : number of similar chunks to retrieve

    Returns:
        Concatenated context string injected into the LLM system prompt.
    """
    #if RAG_BACKEND == "chroma":
    #return _retrieve_chroma(query, top_k)
    #else:  # faiss (default)
    return _retrieve_faiss(query, top_k)


def rebuild_index():
    """
    Force-rebuilds the currently selected vector store index.
    Call this after updating autostream_kb.json.
    """
    #if RAG_BACKEND == "chroma":
    _rebuild_chroma()
    #else:
    #_rebuild_faiss()
    if RAG_BACKEND == "chroma":
        _rebuild_chroma()
    else:
        _rebuild_faiss()


def get_vector_store_info() -> dict:
    """Returns runtime info about the vector store (used by Streamlit UI)."""
    docs = build_documents()
    info = {
        "rag_backend":   RAG_BACKEND,
        "embed_backend": EMBED_BACKEND,
        "embed_model":   (
            EMBED_MODEL if EMBED_BACKEND not in ("tfidf", "fake")
            else EMBED_BACKEND.upper() + " (local)"
        ),
        "top_k":         TOP_K,
        "kb_chunks":     len(docs),
        "faiss_dir":     str(FAISS_DIR),
        "chroma_dir":    str(CHROMA_DIR),
        "langchain_faiss_class":  "langchain_community.vectorstores.FAISS",
        "langchain_chroma_class": "langchain_chroma.Chroma",
    }

    # FAISS stats
    if FAISS_DIR.exists():
        try:
            store = _get_faiss_store()
            info["faiss_vectors"] = store.index.ntotal
            info["faiss_dim"]     = store.index.d
        except Exception:
            info["faiss_vectors"] = "index on disk"

    # Chroma stats
    if CHROMA_DIR.exists():
        try:
            store = _get_chroma_store()
            info["chroma_count"] = store._collection.count()
        except Exception:
            info["chroma_count"] = "collection on disk"

    return info
