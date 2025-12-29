# embed.py
"""
Generate embeddings for document chunks and store them.
"""
import numpy as np
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import config
from ingest import main as ingest_documents

def load_embedding_model():
    """
    Load the sentence transformer model.
    """
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("✓ Model loaded")
    return model

def generate_embeddings(chunks: List[Dict[str, str]], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for all chunks.
    """
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings: np.ndarray, chunks: List[Dict[str, str]]):
    """
    Save embeddings and metadata to disk.
    """
    # Save embeddings as numpy array
    np.save(config.EMBEDDINGS_FILE, embeddings)
    print(f"✓ Saved embeddings to: {config.EMBEDDINGS_FILE}")
    
    # Save metadata as JSON
    metadata = {
        'chunks': chunks,
        'embedding_model': config.EMBEDDING_MODEL,
        'embedding_dim': config.EMBEDDING_DIM,
        'chunk_size': config.CHUNK_SIZE,
        'chunk_overlap': config.CHUNK_OVERLAP,
        'num_chunks': len(chunks)
    }
    
    with open(config.METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved metadata to: {config.METADATA_FILE}")

def main():
    """
    Main embedding pipeline.
    """
    print("=" * 50)
    print("GENERATING EMBEDDINGS")
    print("=" * 50)
    
    # Step 1: Ingest documents
    chunks = ingest_documents()
    
    if not chunks:
        print("\n⚠ No chunks to embed!")
        return
    
    # Step 2: Load embedding model
    model = load_embedding_model()
    
    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks, model)
    
    # Step 4: Save everything
    save_embeddings(embeddings, chunks)
    
    print("\n" + "=" * 50)
    print("✓ EMBEDDING PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Total chunks embedded: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()