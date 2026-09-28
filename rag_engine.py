from dataclasses import asdict
from typing import Iterable

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_MODEL
from ingestion import Chunk

class RAGEngine:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: list[Chunk] = []

    def build_index(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            return
        vectors = self.embedder.encode(
            [item.text for item in self.chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        query_vector = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            item = asdict(self.chunks[index])
            item["score"] = float(score)
            results.append(item)
        return results

    def answer(self, question: str, retrieved: list[dict]) -> str:
        if not retrieved:
            return "I could not find relevant information in the knowledge base."
        if not OPENAI_API_KEY:
            return "Add OPENAI_API_KEY to .env to enable LLM answers.\n\nRetrieved context:\n\n" + "\n\n".join(item["text"] for item in retrieved)
        context = "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in retrieved)
        system_prompt = (
            "You are a grounded knowledge assistant. Answer using only the provided context. "
            "If the context does not contain enough evidence, say the answer is not available "
            "in the knowledge base. Do not invent facts. Keep answers clear and concise."
        )
        user_prompt = f"Knowledge base context:\n{context}\n\nQuestion:\n{question}\n\nAnswer with the strongest evidence from the context. Mention source names when useful."
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
