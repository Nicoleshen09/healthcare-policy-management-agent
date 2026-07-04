"""A deliberately tiny RAG layer.

No FAISS, no Chroma, no local model download: one policy document is small, so an
in-memory numpy cosine search is more than enough and keeps the dependency
surface to numpy + openai. The embedding function is injected, so tests can pass
a deterministic fake and the demo can pass real OpenAI embeddings.
"""

import hashlib
import os
import re

import numpy as np

EMBED_MODEL = os.environ.get("POLICY_AGENT_EMBED_MODEL", "text-embedding-3-small")


def lexical_embed(texts: list, dim: int = 512) -> list:
    """Offline hashing bag-of-words embedding.

    Used by the deterministic demo mode so retrieval runs with no API key and no
    model download. Not as good as real embeddings, but enough for lexical overlap
    on a small policy, and fully reproducible (stable hash).
    """
    vecs = []
    for text in texts:
        v = np.zeros(dim, dtype=float)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            v[idx] += 1.0
        vecs.append(v)
    return vecs


def chunk_policy(text: str) -> list:
    """Split a policy into citable sections (one paragraph = one chunk).

    The citation id follows the document's own numbering ("Section 2") when the
    heading has one, so a cite lines up with what a reader sees in the text.
    Paragraphs without a section number (a title or preamble) get a "P{n}" id.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, p in enumerate(paras):
        heading = p.splitlines()[0].lstrip("# ").strip()[:60]
        m = re.search(r"section\s+(\d+)", heading, re.IGNORECASE)
        cid = f"Section {m.group(1)}" if m else f"P{i + 1}"
        chunks.append({"id": cid, "heading": heading, "text": p})
    return chunks


def openai_embed(texts: list) -> list:
    """Default embedding function using the OpenAI embeddings endpoint."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


class VectorStore:
    def __init__(self, embed_fn=openai_embed):
        self.embed_fn = embed_fn
        self.chunks = []
        self.embeddings = None

    def build(self, chunks: list):
        self.chunks = chunks
        vecs = self.embed_fn([c["text"] for c in chunks])
        self.embeddings = np.asarray(vecs, dtype=float)
        return self

    def search(self, query: str, k: int = 3) -> list:
        q = np.asarray(self.embed_fn([query])[0], dtype=float)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q) + 1e-9
        sims = (self.embeddings @ q) / norms
        order = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in order]
