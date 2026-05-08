import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.vector_rag import search_vector_knowledge


def main():
    questions = [
        "RAG 是什么？",
        "PagedAttention 解决了什么问题？",
        "FastAPI 在大模型应用后端中负责什么？",
        "这个项目有没有接入 Redis？",
    ]

    for question in questions:
        print("=" * 80)
        print(f"Question: {question}")

        results = search_vector_knowledge(question, top_k=2)

        if not results:
            print("No relevant chunks found.")
            continue
        for result in results:
            print("-" * 80)
            print(f"{result['chunk_id']} score={result['score']:.4f}")
            print(result["text"])


if __name__ == "__main__":
    main()