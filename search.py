# search.py
"""
Search for relevant chunks using semantic similarity.
"""
import numpy as np
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import config

class VectorStore:
    """
    Simple vector store for semantic search.
    """
    def __init__(self):
        self.embeddings = None
        self.metadata = None
        self.model = None
    
    def load(self):
        """
        Load embeddings and metadata from disk.
        """
        if not config.EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {config.EMBEDDINGS_FILE}. "
                "Run embed.py first!"
            )
        
        # Load embeddings
        self.embeddings = np.load(config.EMBEDDINGS_FILE)
        print(f"✓ Loaded {self.embeddings.shape[0]} embeddings")
        
        # Load metadata
        with open(config.METADATA_FILE, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"✓ Loaded embedding model: {config.EMBEDDING_MODEL}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for most similar chunks to the query.
        """
        if top_k is None:
            top_k = config.TOP_K
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= config.SIMILARITY_THRESHOLD:
                chunk_data = self.metadata['chunks'][idx].copy()
                chunk_data['similarity'] = float(similarity)
                results.append(chunk_data)
        
        return results
    
    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings.
        """
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Dot product = cosine similarity for normalized vectors
        similarities = np.dot(embeddings_norm, query_norm)
        return similarities

def main():
    """
    Test the search functionality.
    """
    print("=" * 50)
    print("SEMANTIC SEARCH (RAG)")
    print("=" * 50)
    
    store = VectorStore()
    store.load()

    print("\nType a query to search your documents.")
    print("Press ENTER without typing anything to exit.\n")

    while True:
        query = input("🔎 Query: ").strip()
        if not query:
            print("Exiting search.")
            break

        print("-" * 50)
        results = store.search(query)

        if not results:
            print("❌ No results found above similarity threshold.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Similarity: {result['similarity']:.3f}")
                print(f"   Source: {result['source']}")
                print(f"   Chunk: {result['chunk_id'] + 1}/{result['total_chunks']}")
                print(f"   Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()