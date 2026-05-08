import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.rag import search_knowledge


def main():
    questions = [
        "RAG 是什么，为什么能减少幻觉？",
        "PagedAttention 解决了什么问题？",
        "FastAPI 在大模型应用后端里负责什么？",
    ]

    for question in questions:
        print("=" * 80)
        print(f"question: {question}")
        results = search_knowledge(
            query=question,
            file_path="data/knowledge.txt",
            chunk_size=220,
            overlap=40,
            top_k=2,
        )

        for result in results:
            print("-" * 80)
            print(f"{result.chunk_id} score={result.score:.4f}")
            print(result.text)


if __name__ == "__main__":
    main()
