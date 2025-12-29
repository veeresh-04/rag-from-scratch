"""
RAG FROM SCRATCH
Main Application - Interactive Q&A and Summarization
"""

from search import VectorStore
from generate import generate_response
from prompt import create_system_prompt
import config


class RAGApplication:
    """
    End-to-end RAG application.
    Handles:
    - Question Answering
    - Document Summarization
    - Graceful fallbacks
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.system_prompt = create_system_prompt()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def initialize(self) -> bool:
        print("=" * 60)
        print("RAG FROM SCRATCH - Interactive CLI")
        print("=" * 60)

        try:
            self.vector_store.load()
            print("\n✓ Vector store loaded successfully")
            return True

        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            print("\nSetup required:")
            print("  1. Add documents to data/")
            print("  2. Run: python ingest.py")
            print("  3. Run: python embed.py")
            return False

        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            return False

    # --------------------------------------------------
    # Query classification
    # --------------------------------------------------
    @staticmethod
    def is_summarization_query(query: str) -> bool:
        summary_keywords = [
            "summarize", "summary", "summarise",
            "give overview", "overview", "key points"
        ]
        q = query.lower()
        return any(k in q for k in summary_keywords)

    # --------------------------------------------------
    # Core RAG pipeline
    # --------------------------------------------------
    def answer(self, query: str) -> dict:
        print("\n🔍 Searching for relevant context...")

        is_summary = self.is_summarization_query(query)

        # For summarization → retrieve many chunks
        if is_summary:
            top_k = len(self.vector_store.metadata.get("chunks", []))
        else:
            top_k = config.TOP_K

        context_chunks = self.vector_store.search(query, top_k=top_k)

        # No relevant chunks
        if not context_chunks:
            return {
                "query": query,
                "response": (
                    "I couldn't find any relevant information in the "
                    "knowledge base to answer this question."
                ),
                "context_chunks": [],
                "num_chunks": 0
            }

        print(f"✓ Retrieved {len(context_chunks)} chunk(s)")

        print("💭 Generating response...")
        return generate_response(
            query=query,
            context_chunks=context_chunks,
            system_prompt=self.system_prompt
        )

    # --------------------------------------------------
    # Display helpers
    # --------------------------------------------------
    @staticmethod
    def display_result(result: dict):
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["response"])

        if result["num_chunks"] > 0:
            print("\n" + "-" * 60)
            print("SOURCES")
            print("-" * 60)

            for i, chunk in enumerate(result["context_chunks"], 1):
                print(
                    f"\n[{i}] {chunk['source']} "
                    f"(similarity: {chunk['similarity']:.3f})"
                )
                print(f"    {chunk['text'][:200]}...")

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------
    def show_stats(self):
        meta = self.vector_store.metadata

        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE STATS")
        print("=" * 60)
        print(f"Total chunks      : {meta.get('num_chunks', 0)}")
        print(f"Embedding model   : {meta.get('embedding_model', 'N/A')}")
        print(f"Chunk size        : {meta.get('chunk_size', 0)} characters")

    # --------------------------------------------------
    # Interactive loop
    # --------------------------------------------------
    def run(self):
        if not self.initialize():
            return

        print("\n" + "=" * 60)
        print("💬 Ask questions or request summaries")
        print("Commands: help | stats | quit")
        print("=" * 60)

        while True:
            try:
                query = input("\nYou: ").strip()

                if not query:
                    continue

                if query.lower() in {"quit", "exit", "q"}:
                    print("\n👋 Goodbye!")
                    break

                if query.lower() == "help":
                    print("\nExamples:")
                    print(" - What is retrieval augmented generation?")
                    print(" - Why is chunking important in RAG?")
                    print(" - Summarize sample.txt")
                    continue

                if query.lower() == "stats":
                    self.show_stats()
                    continue

                result = self.answer(query)
                self.display_result(result)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
def main():
    app = RAGApplication()
    app.run()


if __name__ == "__main__":
    main()
