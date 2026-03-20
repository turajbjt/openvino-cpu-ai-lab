import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, index_path="models/faiss.index"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.index_path = index_path

        # Try loading existing index
        if os.path.exists(index_path):
            self.load()

    def add_documents(self, texts):
        embeddings = self.embedder.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.text_chunks.extend(texts)

        self.save()

    def search(self, query, k=3):
        if self.index is None:
            return []

        query_vec = self.embedder.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, k)

        results = []
        for i in indices[0]:
            if i < len(self.text_chunks):
                results.append(self.text_chunks[i])

        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

            with open(self.index_path + ".txt", "w") as f:
                for chunk in self.text_chunks:
                    f.write(chunk.replace("\n", " ") + "\n")

    def load(self):
        self.index = faiss.read_index(self.index_path)

        with open(self.index_path + ".txt", "r") as f:
            self.text_chunks = f.readlines()

    def reset(self):
        self.index = None
        self.text_chunks = []

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if os.path.exists(self.index_path + ".txt"):
            os.remove(self.index_path + ".txt")
