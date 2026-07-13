import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fsspec.implementations.local import LocalFileSystem
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.data_loader import (
    MarkdownLoader,
    PdfLoader,
)
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# LLM_BASE_URL 留空時，openai SDK 會用官方 OpenAI endpoint；
# 填了就可以指到 Azure OpenAI / 相容 gateway 等任何 OpenAI-compatible 端點。
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

VECTOR_INDEX_NAME = "chunk_index"

INPUT_DIR = "input"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_driver() -> Driver:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def get_llm(*, json_mode: bool = False) -> OpenAILLM:
    """json_mode=True 給 schema 推斷 / KG 抽取用；最終自然語言回答要 False。"""
    model_params = {"temperature": 0}
    if json_mode:
        model_params["response_format"] = {"type": "json_object"}
    return OpenAILLM(
        model_name=CHAT_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model_params=model_params,
    )


def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def find_input_files(input_dir: str = INPUT_DIR) -> List[str]:
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(
            f"找不到資料夾 '{input_dir}'，請建立這個資料夾並放入要 ingest 的 PDF/TXT/MD"
        )
    files = [
        str(p)
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(
            f"'{input_dir}' 資料夾裡沒有找到支援的檔案（{SUPPORTED_EXTENSIONS}）"
        )
    return sorted(files)


def load_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    fs = LocalFileSystem()
    if suffix == ".pdf":
        return PdfLoader.load_file(file_path, fs)
    if suffix in (".md", ".markdown"):
        return MarkdownLoader.load_file(file_path, fs)
    if suffix == ".txt":
        return Path(file_path).read_text(encoding="utf-8")
    raise ValueError(f"不支援的副檔名: {suffix}")
