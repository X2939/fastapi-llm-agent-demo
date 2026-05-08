import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from app.llm_client import chat_completion

from app.rag import load_and_split_knowledge

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path("data/vector_index.faiss")
CHUNKS_PATH = Path("data/vector_chunks.json")

@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")

def build_vector_index(
    file_path: str = "data/knowledge.txt",
    chunk_size: int = 500,
    overlap: int = 80,
) -> int:
    chunks = load_and_split_knowledge(
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    model = load_embedding_model()
    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")#形状是(知识库文本数量, 向量维度)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    chunk_records = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end,
        }
        for chunk in chunks
    ]

    CHUNKS_PATH.write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )#indent=2格式化 JSON，让文件好看、好阅读不然所有内容挤在一行，没法看。

    return len(chunks)

def load_vector_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Vector index not found: {INDEX_PATH}. Run scripts/build_rag_index.py first."
        )
    
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Vector chunks not found: {CHUNKS_PATH}. Run scripts/build_rag_index.py first."
        )
    
    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks

def search_vector_knowledge(query: str,
                            top_k: int = 3,
                            score_threshold:float = 0.25,
                            ) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty.")
    
    index, chunks = load_vector_index()
    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):#因为 FAISS 支持一次查多个问题,你一次只查一个问题，所以必须取 [0],问题多了就往后排
        if idx == -1:
            continue

        if float(score) < score_threshold:
            continue

        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "text": chunk["text"],
            }
        )

    return results

def build_vector_rag_context(results: list[dict]) -> str:
    if not results:
        return ""

    context_parts = []
    for result in results:
        context_parts.append(
            f"[{result['chunk_id']} score={result['score']:.4f}]\n{result['text']}"
        )

    return "\n\n".join(context_parts)

def has_required_terms(question: str, context: str) -> bool:
    required_terms = [
        "redis",
        "向量数据库",
        "faiss",
        "chroma",
        "milvus",
        "mysql",
        "postgresql",
    ]

    question_lower = question.lower()
    context_lower = context.lower()

    for term in required_terms:
        if term in question_lower and term not in context_lower:
            return False

    return True


def answer_with_vector_rag(
    question: str,
    top_k: int = 3,
    score_threshold: float = 0.25,
) -> str:
    results = search_vector_knowledge(
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    context = build_vector_rag_context(results)

    if not context:
        return "知识库中没有检索到相关资料，无法基于资料回答。"
    if not has_required_terms(question, context):
        return "知识库中没有检索到相关资料，无法基于资料回答。"

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严格的本地知识库问答助手。"
                "你只能使用用户提供的资料回答问题。"
                "禁止使用资料之外的常识、背景知识或推测。"
                "如果资料中没有明确答案，必须回答：无法根据现有资料确定。"
                "回答时不要提到资料中没有出现的模型、产品或例子。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"以下是允许使用的资料：\n{context}\n\n"
                f"用户问题：{question}\n\n"
                "请严格按下面格式回答：\n"
                "1. 结论：用 1-2 句话直接回答用户问题。\n"
                "2. 依据：列出你使用的资料原文要点，只能来自上面的资料。\n"
                "3. 不确定信息：如果资料没有提到，必须明确说明。\n\n"
                "要求：不要使用资料外的背景知识，不要举资料外的例子，不要扩展资料没有说的原因。"
            ),
        },
    ]

    return chat_completion(messages=messages, temperature=0.2)