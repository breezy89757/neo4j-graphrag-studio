"""用自然語言問已建好的知識圖譜，同時把牽涉到的 entity/relationship
拆解成結構化資料，給前端（app.py）畫子圖用。

CLI 用法：
    python query.py "這家公司的主要競爭對手是誰？"
"""

import sys
from typing import Any, Dict, List

import neo4j
from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from common import VECTOR_INDEX_NAME, get_driver, get_embedder, get_llm

# 用 VectorCypherRetriever 而非單純 VectorRetriever：先做向量搜尋定位相關
# chunk，再沿著關係多繞 1-2 hop 抓相關 entity/relationship，這是「圖」相對
# 純向量 RAG 的價值所在。relationship 拆成結構化 map list 回傳，文字（給
# LLM 當 context）跟結構化資料（給前端畫圖）用 result_formatter 分流。
GRAPH_TRAVERSAL_QUERY = """
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[relList:!FROM_CHUNK]-{1,2}(nb)
UNWIND relList AS rel
WITH chunk, collect(DISTINCT rel) AS rels
RETURN
  chunk.text AS chunk_text,
  [r IN rels | {
    source_id: elementId(startNode(r)),
    source_name: coalesce(startNode(r).name, elementId(startNode(r))),
    source_type: labels(startNode(r))[0],
    target_id: elementId(endNode(r)),
    target_name: coalesce(endNode(r).name, elementId(endNode(r))),
    target_type: labels(endNode(r))[0],
    type: type(r)
  }] AS relationships
"""


def _format_result(record: neo4j.Record) -> RetrieverResultItem:
    relationships = record.get("relationships") or []
    lines = [record.get("chunk_text") or ""]
    lines += [
        f"{r['source_name']} -[{r['type']}]-> {r['target_name']}"
        for r in relationships
    ]
    content = "\n".join(line for line in lines if line)
    return RetrieverResultItem(content=content, metadata={"relationships": relationships})


def build_retriever(driver: Driver, embedder: Embedder) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        retrieval_query=GRAPH_TRAVERSAL_QUERY,
        result_formatter=_format_result,
    )


# 抽取出來的 entity name 常是全稱（「Acme Shipping Corporation」），但問答
# 文字多半用簡稱（「Acme Shipping」）。比對前先把常見公司法律形式後綴去
# 掉，用「核心名稱」比對兩種說法才連得起來。由長到短排序避免切得不乾淨。
_CORP_SUFFIXES = [
    "股份有限公司", "有限公司", "股份有限", "股份公司", "公司",
    "corporation", "incorporated", "limited", "co., ltd.", "co.,ltd.",
    "co., ltd", "corp.", "corp", "inc.", "inc", "ltd.", "ltd", "co.",
]


def _core_name(name: str) -> str:
    stripped = name.strip()
    lowered = stripped.lower()
    for suffix in _CORP_SUFFIXES:
        if lowered.endswith(suffix):
            return stripped[: len(stripped) - len(suffix)].strip()
    return stripped


def _filter_to_answer(
    question: str,
    answer: str,
    entities: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """把 retriever 撈回來的整個 1-2 hop 鄰域，縮小成答案裡真的有提到的
    entity/relationship，子圖才會對應到這次回答的內容，而不是整份文件的
    鄰域。比對不到任何 entity，或篩完後完全沒有邊（比對抓歪），就回退成
    完整子圖，避免畫出斷開或空白的圖。
    """
    haystack = f"{question}\n{answer}"
    matched_ids = {
        node_id
        for node_id, e in entities.items()
        if e["name"] and _core_name(e["name"]) in haystack
    }
    if not matched_ids:
        return entities, relationships

    filtered_entities = {nid: e for nid, e in entities.items() if nid in matched_ids}
    filtered_relationships = [
        r
        for r in relationships
        if r["source"] in matched_ids and r["target"] in matched_ids
    ]
    if relationships and not filtered_relationships:
        return entities, relationships
    return filtered_entities, filtered_relationships


def _build_verification_cypher(
    entities: Dict[str, Dict[str, Any]], relationships: List[Dict[str, Any]]
) -> str:
    """組一段可以貼到 Neo4j Browser 驗證的 Cypher，對應畫面上顯示的子圖。"""
    if not entities:
        return ""
    ids = [e["id"] for e in entities.values()]
    ids_literal = ", ".join(f'"{i}"' for i in ids)
    types_literal = ", ".join(f'"{t}"' for t in sorted({r["type"] for r in relationships}))
    type_clause = f"\n  AND type(r) IN [{types_literal}]" if types_literal else ""
    return (
        f"MATCH (a)-[r]-(b)\n"
        f"WHERE elementId(a) IN [{ids_literal}]\n"
        f"  AND elementId(b) IN [{ids_literal}]"
        f"{type_clause}\n"
        f"RETURN a, r, b"
    )


def ask(
    question: str,
    driver: Driver,
    llm: LLMInterface,
    embedder: Embedder,
    top_k: int = 5,
) -> Dict[str, Any]:
    """回傳格式：
        {
            "answer": "自然語言答案",
            "entities": [{"id": ..., "label": ..., "type": "Company", "name": "..."}],
            "relationships": [{"source": ..., "target": ..., "type": "COMPETES_WITH"}],
            "cypher": "MATCH (a)-[r]-(b) WHERE ... RETURN a, r, b",
        }
    """
    retriever = build_retriever(driver, embedder)
    rag = GraphRAG(retriever=retriever, llm=llm)
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )

    entities: Dict[str, Dict[str, Any]] = {}
    relationships: List[Dict[str, Any]] = []
    seen_rels = set()

    items = response.retriever_result.items if response.retriever_result else []
    for item in items:
        for r in (item.metadata or {}).get("relationships", []):
            for node_id, name, node_type in (
                (r["source_id"], r["source_name"], r["source_type"]),
                (r["target_id"], r["target_name"], r["target_type"]),
            ):
                entities.setdefault(
                    node_id,
                    {"id": node_id, "label": name, "type": node_type, "name": name},
                )
            key = (r["source_id"], r["target_id"], r["type"])
            if key not in seen_rels:
                seen_rels.add(key)
                relationships.append(
                    {"source": r["source_id"], "target": r["target_id"], "type": r["type"]}
                )

    entities, relationships = _filter_to_answer(
        question, response.answer, entities, relationships
    )

    return {
        "answer": response.answer,
        "entities": list(entities.values()),
        "relationships": relationships,
        "cypher": _build_verification_cypher(entities, relationships),
    }


def main(query_text: str) -> None:
    driver = get_driver()
    llm = get_llm()
    embedder = get_embedder()

    print(f"問題: {query_text}\n")
    result = ask(query_text, driver, llm, embedder)
    print(f"回答:\n{result['answer']}")

    driver.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "這批文件整體在講什麼？"
    main(q)
