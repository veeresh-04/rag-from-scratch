# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Chunking settings
CHUNK_SIZE = 400  # characters (not tokens for simplicity)
CHUNK_OVERLAP = 50  # characters

# Retrieval settings
TOP_K = 3  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity to include

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2

# LLM settings - Using Ollama
USE_OLLAMA = True  # Set to False to use OpenAI instead
OLLAMA_MODEL = "llama3.2"  # or "mistral", "phi3", "gemma2"
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI settings (fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# Generation settings
TEMPERATURE = 0.1  # Low = more focused answers
MAX_TOKENS = 500

# Storage paths
EMBEDDINGS_DIR = BASE_DIR / "embeddings_store"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"
DATA_FOLDER = BASE_DIR / "data"

# Supported file types
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf']

# Create directories if they don't exist
EMBEDDINGS_DIR.mkdir(exist_ok=True)
DATA_FOLDER.mkdir(exist_ok=True)

# Validation
if not USE_OLLAMA and not OPENAI_API_KEY:
    print("Warning: No API key found and Ollama not enabled. Set OPENAI_API_KEY in .env or enable Ollama.")