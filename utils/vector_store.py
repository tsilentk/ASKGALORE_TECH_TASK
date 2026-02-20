import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class VectorStoreManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.files_metadata = {} # {filename: chunks}
        self.all_chunks = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _rebuild_index(self):
        """Rebuilds the FAISS index from all_chunks."""
        self.all_chunks = []
        for filename in self.files_metadata:
            for chunk in self.files_metadata[filename]:
                self.all_chunks.append(f"Source Document: {filename}\n{chunk}")
        
        if not self.all_chunks:
            self.index = None
            return

        embeddings = np.array(self.model.encode(self.all_chunks)).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension) # Changed to Inner Product (Cosine Similarity because normalized)
        self.index.add(embeddings)

    def add_file(self, filename: str, chunks: list[str]):
        """Adds a new file's chunks to the store and rebuilds/updates index."""
        self.files_metadata[filename] = chunks
        self._rebuild_index()

    def remove_file(self, filename: str):
        """Removes a file and its chunks, then rebuilds the index."""
        if filename in self.files_metadata:
            del self.files_metadata[filename]
            self._rebuild_index()
            return True
        return False

    def clear_all(self):
        """Clears everything."""
        self.files_metadata = {}
        self.all_chunks = []
        self.index = None

    def search(self, query: str, k: int = 4) -> list[str]:
        """Searches the FAISS index."""
        results, _ = self.search_with_scores(query, k)
        return results

    def search_with_scores(self, query: str, k: int = 4) -> tuple[list[str], list[float]]:
        """Searches the FAISS index and returns results and distances."""
        if self.index is None or not self.all_chunks:
            return [], []
            
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Ensure k doesn't exceed total chunks
        k = min(k, len(self.all_chunks))
        distances, indices = self.index.search(query_embedding, k)
        
        results = [self.all_chunks[i] for i in indices[0] if i != -1]
        # For Inner Product (Cosine Similarity), "distances" are actually similarity scores (-1 to 1)
        scores = [float(d) for d in distances[0]]
        return results, scores

    def save(self, path: str = "faiss_index"):
        """Saves metadata and rebuildable state."""
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(self.files_metadata, f)

    def load(self, path: str = "faiss_index"):
        """Loads metadata and rebuilds index."""
        if os.path.exists(f"{path}_meta.pkl"):
            with open(f"{path}_meta.pkl", "rb") as f:
                self.files_metadata = pickle.load(f)
            self._rebuild_index()
            return True
        return False

# Global instance
vector_store = VectorStoreManager()
