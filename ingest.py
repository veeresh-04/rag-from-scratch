# ingest.py
"""
Loads documents from the data/ folder and splits them into chunks.
"""
import os
from pathlib import Path
from typing import List, Dict
import config

def load_documents() -> List[Dict[str, str]]:
    """
    Load all supported documents from the data folder.
    Returns a list of dicts with 'content' and 'source' keys.
    """
    documents = []
    
    if not config.DATA_FOLDER.exists():
        print(f"Data folder not found: {config.DATA_FOLDER}")
        return documents
    
    for file_path in config.DATA_FOLDER.rglob('*'):
        if file_path.is_file() and file_path.suffix in config.SUPPORTED_EXTENSIONS:
            try:
                # Handle different file types
                if file_path.suffix == '.pdf':
                    content = load_pdf(file_path)
                else:  # .txt, .md
                    content = file_path.read_text(encoding='utf-8')
                
                documents.append({
                    'content': content,
                    'source': str(file_path.relative_to(config.DATA_FOLDER))
                })
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

def load_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file.
    Requires: pip install pypdf
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        print("Warning: pypdf not installed. Install with: pip install pypdf")
        return ""

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks

def process_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Split documents into chunks with metadata.
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_text(doc['content'], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': doc['source'],
                'chunk_id': i,
                'total_chunks': len(chunks)
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks

def main():
    """
    Main ingestion pipeline.
    """
    print("=" * 50)
    print("INGESTING DOCUMENTS")
    print("=" * 50)
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("\n⚠ No documents found! Add .txt, .md, or .pdf files to the data/ folder.")
        return []
    
    # Process into chunks
    chunks = process_documents(documents)
    
    print("\n✓ Ingestion complete!")
    return chunks

if __name__ == "__main__":
    chunks = main()
    if chunks:
        print(f"\nExample chunk:\n{chunks[0]['text'][:200]}...")