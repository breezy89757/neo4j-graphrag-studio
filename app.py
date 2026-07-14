"""Streamlit 前端：

    uv run streamlit run app.py

Schema 設定分頁：上傳/選擇文件 -> LLM 推斷 schema -> 使用者勾選調整 -> 建圖
問答分頁：聊天 + 這次查詢牽涉到的 entity/relationship 子圖視覺化
"""

import asyncio
from pathlib import Path

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

import benchmark
import common
from Ingest import ensure_vector_index, ingest_files
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    SchemaBuilder,
    SchemaFromTextExtractor,
)
from query import ask

st.set_page_config(page_title="Neo4j GraphRAG Studio", layout="wide")

SCHEMA_JSON_PATH = "inferred_schema.json"
COLOR_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


def color_for_type(type_name: str) -> str:
    """依類型第一次出現的順序從調色盤指派顏色，避免用 hash 分色時
    不同類型撞到同一個 index、圖上看起來全部同色。"""
    assigned = st.session_state.setdefault("type_color_map", {})
    if type_name not in assigned:
        assigned[type_name] = COLOR_PALETTE[len(assigned) % len(COLOR_PALETTE)]
    return assigned[type_name]


@st.cache_resource
def cached_driver():
    return common.get_driver()


@st.cache_resource
def cached_extraction_llm():
    return common.get_llm(json_mode=True)


@st.cache_resource
def cached_chat_llm():
    return common.get_llm(json_mode=False)


@st.cache_resource
def cached_embedder():
    return common.get_embedder()


def init_state():
    defaults = {
        "messages": [],
        "last_graph": {"entities": [], "relationships": []},
        "last_cypher": "",
        "inferred_schema": None,
        "schema_source_file": None,
        "node_selection": {},
        "rel_selection": {},
        "custom_node_types": [],
        "custom_rel_types": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def pick_input_file(key_prefix: str) -> str | None:
    source = st.radio(
        "文件來源",
        ["上傳新文件", "使用 input/ 資料夾中的既有檔案"],
        key=f"{key_prefix}_source",
        horizontal=True,
    )
    if source == "上傳新文件":
        uploaded = st.file_uploader(
            "上傳 PDF / TXT / MD",
            type=["pdf", "txt", "md"],
            key=f"{key_prefix}_uploader",
        )
        if uploaded is None:
            return None
        dest = Path(common.INPUT_DIR) / uploaded.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded.getvalue())
        st.caption(f"已儲存至 `{dest}`")
        return str(dest)

    try:
        existing_files = common.find_input_files(common.INPUT_DIR)
    except FileNotFoundError as e:
        st.warning(str(e))
        return None
    return st.selectbox("選擇檔案", existing_files, key=f"{key_prefix}_select")


def render_schema_tab():
    st.subheader("Step 1 · 選擇文件")
    file_path = pick_input_file("schema")

    free_mode = st.checkbox(
        "跳過 schema，交給 LLM 自由發揮（FREE 模式）",
        help="不會有 node/relationship type 的引導，LLM 抽取到什麼就寫什麼進圖。",
    )

    if free_mode:
        st.info("FREE 模式：不需要推斷 schema，直接建圖。")
        if st.button("以 FREE 模式建圖", disabled=file_path is None, type="primary"):
            with st.spinner("Ingesting（FREE 模式）..."):
                driver = cached_driver()
                asyncio.run(
                    ingest_files(
                        [file_path],
                        driver,
                        cached_extraction_llm(),
                        cached_embedder(),
                        schema="FREE",
                    )
                )
                ensure_vector_index(driver)
            st.success(f"已完成建圖：{file_path}")
        return

    st.subheader("Step 2 · 推斷 Schema")
    if st.button("推斷 Schema", disabled=file_path is None):
        with st.spinner("讀取文件並呼叫 LLM 推斷 schema..."):
            text = common.load_text(file_path)
            extractor = SchemaFromTextExtractor(llm=cached_extraction_llm())
            inferred = asyncio.run(extractor.run(text=text))
            inferred.save(SCHEMA_JSON_PATH, overwrite=True)

            st.session_state["inferred_schema"] = inferred
            st.session_state["schema_source_file"] = file_path
            st.session_state["node_selection"] = {nt.label: True for nt in inferred.node_types}
            st.session_state["rel_selection"] = {rt.label: True for rt in inferred.relationship_types}
            st.session_state["custom_node_types"] = []
            st.session_state["custom_rel_types"] = []
        st.success(f"推斷完成，結果已存到 `{SCHEMA_JSON_PATH}`")

    inferred: GraphSchema | None = st.session_state["inferred_schema"]
    if inferred is None:
        return

    st.subheader("Step 3 · 確認 / 調整 Schema")
    st.caption(f"來源文件：`{st.session_state['schema_source_file']}`")

    col_node, col_rel = st.columns(2)

    with col_node:
        st.markdown("**Node types**")
        for nt in inferred.node_types:
            st.session_state["node_selection"][nt.label] = st.checkbox(
                nt.label,
                value=st.session_state["node_selection"].get(nt.label, True),
                key=f"node_cb_{nt.label}",
            )
        for label in st.session_state["custom_node_types"]:
            st.session_state["node_selection"][label] = st.checkbox(
                f"{label}（自訂）",
                value=st.session_state["node_selection"].get(label, True),
                key=f"node_cb_custom_{label}",
            )
        with st.form("add_node_type_form", clear_on_submit=True):
            new_node = st.text_input("新增自訂 node type")
            if st.form_submit_button("新增 Node Type") and new_node.strip():
                label = new_node.strip()
                if label not in st.session_state["custom_node_types"]:
                    st.session_state["custom_node_types"].append(label)
                    st.session_state["node_selection"][label] = True
                st.rerun()

    with col_rel:
        st.markdown("**Relationship types**")
        for rt in inferred.relationship_types:
            st.session_state["rel_selection"][rt.label] = st.checkbox(
                rt.label,
                value=st.session_state["rel_selection"].get(rt.label, True),
                key=f"rel_cb_{rt.label}",
            )
        for label in st.session_state["custom_rel_types"]:
            st.session_state["rel_selection"][label] = st.checkbox(
                f"{label}（自訂）",
                value=st.session_state["rel_selection"].get(label, True),
                key=f"rel_cb_custom_{label}",
            )
        with st.form("add_rel_type_form", clear_on_submit=True):
            new_rel = st.text_input("新增自訂 relationship type")
            if st.form_submit_button("新增 Relationship Type") and new_rel.strip():
                label = new_rel.strip()
                if label not in st.session_state["custom_rel_types"]:
                    st.session_state["custom_rel_types"].append(label)
                    st.session_state["rel_selection"][label] = True
                st.rerun()

    st.divider()
    if st.button("確認並建圖", type="primary"):
        selected_nodes = [
            nt for nt in inferred.node_types
            if st.session_state["node_selection"].get(nt.label)
        ] + [
            label for label in st.session_state["custom_node_types"]
            if st.session_state["node_selection"].get(label)
        ]
        selected_rels = [
            rt for rt in inferred.relationship_types
            if st.session_state["rel_selection"].get(rt.label)
        ] + [
            label for label in st.session_state["custom_rel_types"]
            if st.session_state["rel_selection"].get(label)
        ]

        if not selected_nodes:
            st.error("至少要選一個 node type 才能建圖。")
            return

        final_schema = SchemaBuilder.create_schema_model(
            node_types=selected_nodes,
            relationship_types=selected_rels,
        )

        with st.spinner("Ingesting..."):
            driver = cached_driver()
            asyncio.run(
                ingest_files(
                    [st.session_state["schema_source_file"]],
                    driver,
                    cached_extraction_llm(),
                    cached_embedder(),
                    schema=final_schema,
                )
            )
            ensure_vector_index(driver)
        st.success(f"已完成建圖：{st.session_state['schema_source_file']}")


def render_chat_tab():
    col_chat, col_graph = st.columns([0.45, 0.55])

    with col_chat:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("問一個問題...")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.spinner("思考中..."):
                result = ask(
                    question,
                    cached_driver(),
                    cached_chat_llm(),
                    cached_embedder(),
                )
            st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})
            st.session_state["last_graph"] = {
                "entities": result["entities"],
                "relationships": result["relationships"],
            }
            st.session_state["last_cypher"] = result["cypher"]
            st.rerun()

    with col_graph:
        st.subheader("這次查詢的關聯子圖")
        graph_data = st.session_state["last_graph"]
        if not graph_data["entities"]:
            st.info("問一個問題，這裡會顯示牽涉到的 entity / relationship。")
        else:
            nodes = [
                Node(
                    id=e["id"],
                    label=e["name"],
                    title=e["type"],
                    color=color_for_type(e["type"]),
                    size=25,
                )
                for e in graph_data["entities"]
            ]
            edges = [
                Edge(source=r["source"], target=r["target"], label=r["type"])
                for r in graph_data["relationships"]
            ]
            config = Config(
                width="100%",
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
            )
            agraph(nodes=nodes, edges=edges, config=config)

        if st.session_state["last_cypher"]:
            with st.expander("複製 Cypher 到 Neo4j Browser 驗證這張子圖"):
                st.code(st.session_state["last_cypher"], language="cypher")


def render_benchmark_tab():
    st.subheader("同一份文件 x 同一個問題：三種做法對照")
    st.caption(
        "文件需先在「Schema 設定與建圖」分頁完成建圖，這裡才撈得到向量索引與圖譜資料。"
    )

    try:
        existing_files = common.find_input_files(common.INPUT_DIR)
    except FileNotFoundError as e:
        st.warning(str(e))
        return

    file_path = st.selectbox("選擇已建圖的文件", existing_files, key="benchmark_file")
    top_k = st.slider(
        "Top-K（向量檢索片段數，GraphRAG 與純向量 RAG 共用）",
        min_value=1, max_value=10, value=5, key="benchmark_top_k",
    )
    question = st.text_input("輸入問題", key="benchmark_question")

    if st.button("執行基準測試", type="primary", disabled=not question):
        driver = cached_driver()
        llm = cached_chat_llm()
        embedder = cached_embedder()

        with st.spinner("三種做法都在跑，請稍候..."):
            st.session_state["benchmark_results"] = {
                "question": question,
                "file_path": file_path,
                "graphrag": benchmark.ask_graphrag(question, driver, llm, embedder, top_k=top_k),
                "vector_rag": benchmark.ask_vector_rag(question, driver, llm, embedder, top_k=top_k),
                "full_document": benchmark.ask_full_document(question, file_path, llm),
            }

    results = st.session_state.get("benchmark_results")
    if not results:
        st.info("選好文件、輸入問題後按下「執行基準測試」，這裡會並排顯示三種做法的回答。")
        return

    st.divider()
    st.caption(f"問題：{results['question']}　|　文件：`{results['file_path']}`")

    columns = st.columns(3)
    panels = [
        (columns[0], "🕸️ Neo4j GraphRAG（現有架構）", "graphrag"),
        (columns[1], "🧮 純向量 RAG", "vector_rag"),
        (columns[2], "📄 整份文件丟 LLM", "full_document"),
    ]
    for col, label, key in panels:
        with col:
            st.markdown(f"**{label}**")
            result = results[key]
            m1, m2 = st.columns(2)
            m1.metric("耗時", f"{result['elapsed_s']:.1f} s")
            m2.metric("餵給 LLM 的字數", f"{result['context_chars']:,}")
            if result.get("truncated"):
                st.warning(f"文件超過 {benchmark.MAX_FULL_DOC_CHARS:,} 字，已截斷。")
            st.markdown(result["answer"])
            with st.expander(f"查看餵給 LLM 的內容（共 {len(result['context_items'])} 段）"):
                for i, chunk in enumerate(result["context_items"], start=1):
                    st.text_area(
                        f"片段 {i}", chunk, height=150,
                        key=f"benchmark_{key}_chunk_{i}", disabled=True,
                    )


def render_sidebar():
    with st.sidebar:
        st.header("圖譜統計")
        try:
            driver = cached_driver()
            with driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            st.metric("Node 數量", node_count)
            st.metric("Relationship 數量", rel_count)
        except Exception as e:
            st.error(f"無法連線 Neo4j：{e}")


def main():
    render_sidebar()
    tab_chat, tab_schema, tab_benchmark = st.tabs(
        ["💬 問答", "🗂️ Schema 設定與建圖", "🧪 基準測試"]
    )
    with tab_chat:
        render_chat_tab()
    with tab_schema:
        render_schema_tab()
    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
