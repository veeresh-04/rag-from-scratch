# prompt.py
"""
Create prompts for the LLM using retrieved context.
"""
from typing import List, Dict

def create_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Create a RAG prompt with query and retrieved context.
    """
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get('source', 'Unknown')
        text = chunk.get('text', '')
        context_parts.append(f"[{i}] Source: {source}\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Create the full prompt
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and accurate
- Cite which source number [1], [2], etc. you're using when relevant

Answer:"""
    
    return prompt

def create_system_prompt() -> str:
    """
    Create a system prompt for the LLM.
    """
    return """You are a precise and helpful AI assistant. You answer questions based strictly on the provided context. If the context doesn't contain the answer, you honestly say so rather than making up information."""

def main():
    """
    Test prompt creation.
    """
    test_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'source': 'ml_basics.txt',
            'similarity': 0.85
        }
    ]
    
    test_query = "What is machine learning?"
    
    print("=" * 50)
    print("PROMPT TEMPLATE")
    print("=" * 50)
    
    prompt = create_rag_prompt(test_query, test_chunks)
    print(prompt)

if __name__ == "__main__":
    main()