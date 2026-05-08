import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.vector_rag import answer_with_vector_rag


def main():
    questions = [
        "RAG 是什么？它为什么能减少幻觉？",
        "PagedAttention 解决了什么问题？",
        "FastAPI 在大模型应用后端中负责什么？",
        "这个项目有没有接入 Redis？",
        "RAG 是否使用了向量数据库？",
    ]

    for question in questions:
        print("=" * 80)
        print(f"Question: {question}")
        answer = answer_with_vector_rag(question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
