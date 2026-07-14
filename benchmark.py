"""同一份文件、同一個問題，比較三種做法的回答品質與成本：

    1. ask_graphrag       - 目前架構：VectorCypherRetriever（向量定位 chunk + 1-2 hop 圖遍歷）
    2. ask_vector_rag     - 對照組：純 VectorRetriever，只用 chunk embedding，不碰圖
    3. ask_full_document  - 對照組：整份文件塞進 prompt，不做任何檢索

三者回傳同樣的 dict 形狀，方便 app.py 並排顯示：
    {"answer": str, "elapsed_s": float, "context_chars": int, "context_items": List[str]}
"""

import time
from typing import Any, Dict, List

from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

import common
import query

# 全文塞進 prompt 的字數上限。超過就截斷並在結果標記 truncated=True，
# 讓使用者看得出來「整份文件丟 LLM」這個做法在大文件上本來就會失真，
# 而不是誤以為三種做法在同等條件下比較。
MAX_FULL_DOC_CHARS = 60_000


def ask_graphrag(
    question: str,
    driver: Driver,
    llm: LLMInterface,
    embedder: Embedder,
    top_k: int = 5,
) -> Dict[str, Any]:
    retriever = query.build_retriever(driver, embedder)
    rag = GraphRAG(retriever=retriever, llm=llm)

    start = time.perf_counter()
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    elapsed = time.perf_counter() - start

    items = response.retriever_result.items if response.retriever_result else []
    context_items = [item.content for item in items]
    return {
        "answer": response.answer,
        "elapsed_s": elapsed,
        "context_chars": sum(len(c) for c in context_items),
        "context_items": context_items,
    }


def _plain_chunk_formatter(record) -> RetrieverResultItem:
    node = record.get("node")
    text = node.get("text", "") if node else ""
    return RetrieverResultItem(content=text, metadata={"score": record.get("score")})


def build_vector_retriever(driver: Driver, embedder: Embedder) -> VectorRetriever:
    return VectorRetriever(
        driver,
        index_name=common.VECTOR_INDEX_NAME,
        embedder=embedder,
        result_formatter=_plain_chunk_formatter,
    )


def ask_vector_rag(
    question: str,
    driver: Driver,
    llm: LLMInterface,
    embedder: Embedder,
    top_k: int = 5,
) -> Dict[str, Any]:
    retriever = build_vector_retriever(driver, embedder)
    rag = GraphRAG(retriever=retriever, llm=llm)

    start = time.perf_counter()
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    elapsed = time.perf_counter() - start

    items = response.retriever_result.items if response.retriever_result else []
    context_items = [item.content for item in items]
    return {
        "answer": response.answer,
        "elapsed_s": elapsed,
        "context_chars": sum(len(c) for c in context_items),
        "context_items": context_items,
    }


def ask_full_document(
    question: str,
    file_path: str,
    llm: LLMInterface,
    max_chars: int = MAX_FULL_DOC_CHARS,
) -> Dict[str, Any]:
    text = common.load_text(file_path)
    truncated = len(text) > max_chars
    doc_text = text[:max_chars] if truncated else text

    prompt = (
        "根據以下文件內容回答問題。如果文件沒有提到相關資訊，請直接說明找不到。\n\n"
        f"文件內容：\n{doc_text}\n\n"
        f"問題：{question}"
    )

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    return {
        "answer": response.content,
        "elapsed_s": elapsed,
        "context_chars": len(doc_text),
        "context_items": [doc_text],
        "truncated": truncated,
    }
