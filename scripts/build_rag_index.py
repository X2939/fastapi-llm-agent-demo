import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.vector_rag import build_vector_index


def main():
    count = build_vector_index(
        file_path="data/knowledge.txt",
        chunk_size=220,
        overlap=40,
    )
    print(f"built vector index with {count} chunks")


if __name__ == "__main__":
    main()
