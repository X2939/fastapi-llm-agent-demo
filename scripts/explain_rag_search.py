import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.rag import _token_overlap_score, _tokenize, load_and_split_knowledge


def main():
    question = "RAG 是什么，为什么能减少幻觉？"
    query_tokens = _tokenize(question)
    chunks = load_and_split_knowledge(
        file_path="data/knowledge.txt",
        chunk_size=260,
        overlap=40,
    )

    print(f"question: {question}")
    print(f"query_tokens: {sorted(query_tokens)}")
    print()

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        matched_tokens = sorted(query_tokens & chunk_tokens)
        score = _token_overlap_score(query_tokens, chunk_tokens)

        print("=" * 80)
        print(f"{chunk.chunk_id} score={score:.4f}")
        print(f"matched_tokens: {matched_tokens}")
        print(chunk.text)


if __name__ == "__main__":
    main()
