from dataclasses import dataclass
import math
from pathlib import Path
import re
from app.llm_client import chat_completion


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    score: float


def load_text_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {file_path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Knowledge file is empty: {file_path}")

    return text


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    if paragraphs:
        return _split_paragraphs(paragraphs, chunk_size=chunk_size, overlap=overlap)

    return _split_by_characters(text=text, chunk_size=chunk_size, overlap=overlap)


def _split_by_characters(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk-{chunk_index}",
                    text=chunk_text,
                    start=start,
                    end=end,
                )
            )
            chunk_index += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks


def _split_paragraphs(
    paragraphs: list[str],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_length = 0
    search_start = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_parts:
                chunks.append(_build_chunk(chunks, "\n\n".join(current_parts), search_start))
                current_parts = []
                current_length = 0

            chunks.extend(
                _split_by_characters(
                    text=paragraph,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
            search_start += len(paragraph) + 2
            continue

        next_length = current_length + len(paragraph) + (2 if current_parts else 0)#如果把当前这段加进篮子，总长度会是多少？
        if current_parts and next_length > chunk_size:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(_build_chunk(chunks, chunk_text, search_start))

            current_parts = [paragraph]
            current_length = len(paragraph)
            search_start += len(chunk_text) + 2
        else:
            current_parts.append(paragraph)
            current_length = next_length

    if current_parts:
        chunks.append(_build_chunk(chunks, "\n\n".join(current_parts), search_start))

    return chunks


def _build_chunk(existing_chunks: list[TextChunk], text: str, start: int) -> TextChunk:
    return TextChunk(
        chunk_id=f"chunk-{len(existing_chunks) + 1}",
        text=text.strip(),
        start=start,
        end=start + len(text),
    )



def load_and_split_knowledge(
    file_path: str = "data/knowledge.txt",
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[TextChunk]:
    text = load_text_file(file_path)
    return split_text(text=text, chunk_size=chunk_size, overlap=overlap)


def search_knowledge(
    query: str,
    file_path: str = "data/knowledge.txt",
    chunk_size: int = 500,
    overlap: int = 80,
    top_k: int = 3,
) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty.")

    chunks = load_and_split_knowledge(
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    query_tokens = _tokenize(query)
    scored_results: list[SearchResult] = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        score = _token_overlap_score(query_tokens, chunk_tokens)
        if score > 0:
            scored_results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                )
            )

    scored_results.sort(key=lambda result: result.score, reverse=True)
    return scored_results[:top_k]

#分词器
def _tokenize(text: str) -> set[str]:
    lowered = text.lower()
    english_words = re.findall(r"[a-z0-9_./-]+", lowered)
    chinese_terms: list[str] = []

    for chinese_text in re.findall(r"[\u4e00-\u9fff]+", lowered):
        chinese_terms.append(chinese_text)
        chinese_terms.extend(
            chinese_text[index : index + 2]
            for index in range(len(chinese_text) - 1)
        )

    return set(english_words + chinese_terms)


def _token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0

    intersection = left & right
    return len(intersection) / math.sqrt(len(left) * len(right))

def build_rag_context(results: list[SearchResult]) -> str:
    if not results:
        return ""

    context_parts = []
    for result in results:
        context_parts.append(
            f"[{result.chunk_id} score={result.score:.4f}]\n{result.text}"
        )

    return "\n\n".join(context_parts)


def answer_with_rag(
    question: str,
    file_path: str = "data/knowledge.txt",
    top_k: int = 3,
) -> str:
    results = search_knowledge(
        query=question,
        file_path=file_path,
        top_k=top_k,
    )

    context = build_rag_context(results)

    if not context:
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
