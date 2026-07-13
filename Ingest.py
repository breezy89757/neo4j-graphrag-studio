"""把 PDF/文字檔轉成知識圖譜寫入 Neo4j。

`ingest_files()` 的 schema 由呼叫端動態傳入（例如前端 schema 推斷 +
使用者確認調整後的結果），或傳 "FREE" 讓 LLM 自由抽取。

CLI 模式（未指定 schema）用下面的預設 schema：
    python Ingest.py
"""

import asyncio
from typing import List, Literal, Optional, Sequence, Union

from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.schema import GraphSchema, SchemaBuilder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import LLMInterface

from common import (
    EMBEDDING_DIM,
    INPUT_DIR,
    VECTOR_INDEX_NAME,
    find_input_files,
    get_driver,
    get_embedder,
    get_llm,
    load_text,
)

SchemaArg = Optional[Union[GraphSchema, Literal["FREE"]]]

DEFAULT_ENTITY_TYPES = ["Company", "Industry", "Person", "Regulation", "Product"]
DEFAULT_RELATION_TYPES = [
    "COMPETES_WITH",
    "SUBSIDIARY_OF",
    "REGULATED_BY",
    "SUPPLIES_TO",
    "INVESTS_IN",
    "PARTNERS_WITH",
]


def build_kg_pipeline(
    llm: LLMInterface,
    driver: Driver,
    embedder: Embedder,
    *,
    from_file: bool,
    schema: SchemaArg = None,
) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        from_file=from_file,
    )


async def ingest_one(
    kg_builder: SimpleKGPipeline, file_path: str, text: Optional[str] = None
):
    print(f"開始 ingestion: {file_path}")
    try:
        if text is not None:
            result = await kg_builder.run_async(text=text)
        else:
            result = await kg_builder.run_async(file_path=file_path)
        print(f"完成: {file_path}")
        return result
    except Exception as e:
        print(f"失敗: {file_path} — {e}")
        return None


async def ingest_files(
    files: Sequence[str],
    driver: Driver,
    llm: LLMInterface,
    embedder: Embedder,
    schema: SchemaArg = None,
) -> List[object]:
    pdf_builder = build_kg_pipeline(llm, driver, embedder, from_file=True, schema=schema)
    text_builder = build_kg_pipeline(llm, driver, embedder, from_file=False, schema=schema)

    results = []
    for file_path in files:
        is_pdf = file_path.lower().endswith(".pdf")
        builder = pdf_builder if is_pdf else text_builder
        text = None if is_pdf else load_text(file_path)
        results.append(await ingest_one(builder, file_path, text=text))
    return results


def ensure_vector_index(driver: Driver, dimensions: int = EMBEDDING_DIM) -> None:
    try:
        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
        )
        print("向量索引建立完成")
    except Exception as e:
        print(f"向量索引可能已存在，略過: {e}")


async def main(input_dir: str = INPUT_DIR) -> None:
    files = find_input_files(input_dir)
    print(f"找到 {len(files)} 份文件：")
    for f in files:
        print(f"   - {f}")
    print()

    driver = get_driver()
    print("Neo4j 連線成功\n")

    llm = get_llm(json_mode=True)
    embedder = get_embedder()

    schema = SchemaBuilder.create_schema_model(
        node_types=DEFAULT_ENTITY_TYPES,
        relationship_types=DEFAULT_RELATION_TYPES,
    )

    await ingest_files(files, driver, llm, embedder, schema=schema)
    ensure_vector_index(driver)

    driver.close()
    print("\n全部處理完成")


if __name__ == "__main__":
    asyncio.run(main(INPUT_DIR))
