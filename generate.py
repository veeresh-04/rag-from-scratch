# generate.py
"""
Generate responses using LLM (Ollama or OpenAI).
"""
import requests
import json
from typing import List, Dict
import config

class LLMClient:
    """
    Unified client for Ollama and OpenAI.
    """
    def __init__(self):
        self.use_ollama = config.USE_OLLAMA
        
        if self.use_ollama:
            self.model = config.OLLAMA_MODEL
            self.base_url = config.OLLAMA_BASE_URL
            print(f"Using Ollama with model: {self.model}")
        else:
            self.model = config.OPENAI_MODEL
            self.api_key = config.OPENAI_API_KEY
            print(f"Using OpenAI with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response from the LLM.
        """
        if self.use_ollama:
            return self._generate_ollama(prompt, system_prompt)
        else:
            return self._generate_openai(prompt, system_prompt)
    
    def _generate_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using Ollama API.
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
        
        except requests.exceptions.ConnectionError:
            return "❌ Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)"
        except requests.exceptions.Timeout:
            return "❌ Error: Request timed out. Try a smaller document or increase timeout."
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def _generate_openai(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using OpenAI API.
        """
        try:
            import openai
            openai.api_key = self.api_key
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
        
        except ImportError:
            return "❌ Error: openai package not installed. Run: pip install openai"
        except Exception as e:
            return f"❌ Error: {str(e)}"

def generate_response(query: str, context_chunks: List[Dict], system_prompt: str = None) -> Dict:
    """
    Generate a response using RAG pipeline.
    """
    from prompt import create_rag_prompt
    
    # Create prompt with context
    prompt = create_rag_prompt(query, context_chunks)
    
    # Generate response
    client = LLMClient()
    response = client.generate(prompt, system_prompt)
    
    return {
        'query': query,
        'response': response,
        'context_chunks': context_chunks,
        'num_chunks': len(context_chunks)
    }

def main():
    """
    Test LLM generation.
    """
    print("=" * 50)
    print("TESTING LLM GENERATION")
    print("=" * 50)
    
    test_chunks = [
        {
            'text': 'Python is a high-level programming language known for its simplicity and readability.',
            'source': 'python_intro.txt',
            'similarity': 0.9
        }
    ]
    
    test_query = "What is Python?"
    
    print(f"\nQuery: {test_query}\n")
    
    result = generate_response(test_query, test_chunks)
    
    print("Response:")
    print("-" * 50)
    print(result['response'])

if __name__ == "__main__":
    main()