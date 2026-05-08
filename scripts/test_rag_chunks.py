import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.rag import load_and_split_knowledge


def main():
    chunks = load_and_split_knowledge(
        file_path="data/knowledge.txt",
        chunk_size=220,
        overlap=40,
    )

    print(f"total_chunks={len(chunks)}")
    for chunk in chunks:
        print("=" * 80)
        print(f"{chunk.chunk_id} start={chunk.start} end={chunk.end}")
        print(chunk.text)


if __name__ == "__main__":
    main()
