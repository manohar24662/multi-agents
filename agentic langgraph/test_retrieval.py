import sys

from agentic_langgraph.RAG_pipline.services.retrieval.qdrant_store import retrieve_chunks

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunks = retrieve_chunks("Tell me about Durga Manohar")
    for chunk in chunks:
        print("SCORE:", chunk.score)
        print("SOURCE:", chunk.source)
        print("TEXT:", chunk.text)
        print("-" * 80)


if __name__ == "__main__":
    main()
