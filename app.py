import os
import sys
import json
import streamlit as st
import pandas as pd

# 確保專案根目錄在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR, QA_PAIRS_PATH, GOOGLE_API_KEY, ENABLE_RERANKER, ENABLE_GROUNDING_CHECK
from src.pdf_parser import PDFParser
from src.chunking import SmartChunker
from src.vectorstore import VectorStoreManager
from src.retriever import HybridRetriever
from src.chain import RAGChain
from src.evaluator import Evaluator


def get_llm():
    """取得 Gemini LLM"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from config import LLM_MODEL
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                progress_callback=None):
    """PDF 處理 Pipeline：解析 → TOC 解析 → 分塊 → 實體標記 → 建索引"""
    from src.toc_parser import TOCParser
    from src.metadata_enricher import MetadataEnricher

    parser = PDFParser()
    chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs_manager = VectorStoreManager()

    # Step 1: 解析 PDF（含 Vision fallback、header/footer 移除）
    if progress_callback:
        progress_callback(0.15, "解析 PDF 中...")
    documents = parser.parse(pdf_path)

    # Step 2: 解析目錄結構
    if progress_callback:
        progress_callback(0.25, "解析目錄結構中...")
    toc_parser = TOCParser()
    toc_parser.parse(pdf_path)

    # Step 3: 結構感知分塊（注入 TOC 解析器）
    if progress_callback:
        progress_callback(0.35, "文件分塊中...")
    chunker.set_toc_parser(toc_parser)
    chunks, parent_docs = chunker.chunk_documents(documents)

    # Step 4: 實體感知 metadata 標記
    if progress_callback:
        progress_callback(0.45, "實體標記中...")
    enricher = MetadataEnricher()
    company = enricher.detect_company(documents)
    chunks = enricher.enrich_documents(chunks, toc_parser)

    # Step 5: 建立文字向量索引
    if progress_callback:
        progress_callback(0.6, "建立向量索引中...")
    vectorstore = vs_manager.build_index(chunks, pdf_path)

    # Step 6: 建立頁面圖片索引（如可用）
    image_index = None
    try:
        if progress_callback:
            progress_callback(0.8, "建立圖片索引中...")
        image_index = vs_manager.build_image_index(pdf_path)
    except Exception as e:
        print(f"[Warning] 圖片索引建立失敗（非致命）: {e}")

    if progress_callback:
        progress_callback(1.0, "完成！")

    return vectorstore, chunks, parent_docs, company, image_index


def build_rag_chain(vectorstore, chunks, parent_docs, company=None, image_index=None,
                    companies=None):
    """建構完整的 RAG Chain（含 LLM Reranker + Entity Awareness + Grounding Check）"""
    llm = get_llm()

    # 初始化 LLM Reranker
    reranker = None
    if ENABLE_RERANKER:
        try:
            from src.reranker import LLMReranker
            reranker = LLMReranker(llm)
        except Exception as e:
            import streamlit as st
            st.warning(f"Reranker 載入失敗，使用原始排序：{e}")

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=chunks,
        parent_docs=parent_docs,
        reranker=reranker,
        image_index=image_index,
    )
    chain = RAGChain(llm, retriever, vectorstore=vectorstore,
                     current_company=company, current_companies=companies)
    return chain, llm


def list_available_documents():
    """列出已上傳的 PDF 文件"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
    return files


def load_qa_pairs():
    """載入問答集"""
    if os.path.exists(QA_PAIRS_PATH):
        with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    st.set_page_config(
        page_title="年報 RAG 查詢系統",
        page_icon="📊",
        layout="wide"
    )
    st.title("📊 年報智慧問答系統")
    st.caption("上傳任意 PDF 年報，系統自動建立索引並提供智慧問答")

    # ===== 初始化 session state =====
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "current_docs" not in st.session_state:
        st.session_state.current_docs = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None

    # ===== 側邊欄 =====
    with st.sidebar:
        st.header("📄 文件管理")

        # API Key 設定
        api_key = st.text_input(
            "Google API Key",
            value=GOOGLE_API_KEY,
            type="password",
            help="輸入 Gemini API Key"
        )
        if api_key and api_key != "your_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            import config
            config.GOOGLE_API_KEY = api_key

        st.divider()

        # PDF 上傳
        uploaded_files = st.file_uploader(
            "上傳 PDF 年報",
            type=["pdf"],
            accept_multiple_files=True,
            help="支援任何 PDF 檔案，系統會自動解析並建立索引"
        )

        if uploaded_files:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for file in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, file.name)
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

        # 選擇查詢文件（支援多選）
        available_docs = list_available_documents()
        if available_docs:
            selected_docs = st.multiselect(
                "選擇查詢文件（可多選）",
                available_docs,
                default=[available_docs[0]] if available_docs else [],
            )
        else:
            selected_docs = []
            st.info("請先上傳 PDF 文件")

        st.divider()
        st.header("⚙️ 進階設定")
        chunk_size = st.slider("Chunk 大小", 300, 2000, 1000, 50)
        chunk_overlap = st.slider("Chunk 重疊", 50, 500, 200, 10)
        retrieve_k = st.slider("初始檢索數", 10, 50, 20, help="FAISS + BM25 各取此數量進行 RRF 融合")
        rerank_top_k = st.slider("Rerank 後保留數", 3, 10, 5, help="Reranking 後送入 LLM 的文件數")

        st.divider()
        # 建立/載入索引（支援多文件合併）
        if selected_docs:
            vs_manager = VectorStoreManager()

            if st.button("🔨 建立/載入索引", type="primary", width="stretch"):
                import numpy as np
                from src.toc_parser import TOCParser
                from src.metadata_enricher import MetadataEnricher

                all_chunks = []
                all_parent_docs = {}
                all_vectorstores = []
                all_image_vectors = []
                all_image_pages = []
                all_image_sources = []
                all_companies = []

                progress_bar = st.progress(0)
                status_text = st.empty()
                total = len(selected_docs)

                for idx, doc_name in enumerate(selected_docs):
                    pdf_path = os.path.join(UPLOAD_DIR, doc_name)
                    base_pct = idx / total

                    if vs_manager.index_exists(pdf_path):
                        status_text.text(f"載入 {doc_name} 的索引...")
                        progress_bar.progress(base_pct + 0.5 / total)

                        vectorstore = vs_manager.load_index(pdf_path)
                        parser = PDFParser()
                        documents = parser.parse(pdf_path)

                        toc_parser = TOCParser()
                        toc_parser.parse(pdf_path)

                        chunker = SmartChunker(chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap)
                        chunker.set_toc_parser(toc_parser)
                        chunks, parent_docs = chunker.chunk_documents(documents)

                        enricher = MetadataEnricher()
                        company = enricher.detect_company(documents)
                        chunks = enricher.enrich_documents(chunks, toc_parser)

                        image_index = None
                        try:
                            image_index = vs_manager.build_image_index(pdf_path)
                        except Exception:
                            pass
                    else:
                        def make_progress_cb(base, step):
                            def cb(pct, msg=""):
                                progress_bar.progress(min(base + step * pct, 1.0))
                                if msg:
                                    status_text.text(f"{doc_name}: {msg}")
                            return cb

                        vectorstore, chunks, parent_docs, company, image_index = process_pdf(
                            pdf_path, chunk_size, chunk_overlap,
                            make_progress_cb(base_pct, 1.0 / total)
                        )

                    all_vectorstores.append(vectorstore)
                    all_chunks.extend(chunks)
                    all_parent_docs.update(parent_docs)
                    all_companies.append(company)

                    # 收集 image_index 資料
                    if image_index and image_index.get("vectors") is not None:
                        all_image_vectors.append(image_index["vectors"])
                        all_image_pages.extend(image_index["page_numbers"])
                        all_image_sources.extend([doc_name] * len(image_index["page_numbers"]))

                # 合併 FAISS 索引
                merged_vs = all_vectorstores[0]
                for vs in all_vectorstores[1:]:
                    merged_vs.merge_from(vs)

                # 合併 image_index
                merged_image_index = None
                if all_image_vectors:
                    merged_image_index = {
                        "vectors": np.concatenate(all_image_vectors, axis=0),
                        "page_numbers": all_image_pages,
                        "sources": all_image_sources,
                    }

                # 重新編號 chunk_id（避免多文件衝突）
                for i, chunk in enumerate(all_chunks):
                    chunk.metadata["chunk_id"] = f"chunk_{i}"

                # 取得所有公司名稱（用於 scope check）
                valid_companies = [c for c in all_companies if c and c != "unknown"]
                primary_company = valid_companies[0] if valid_companies else None

                chain, llm = build_rag_chain(
                    merged_vs, all_chunks, all_parent_docs,
                    primary_company, merged_image_index,
                    companies=valid_companies,
                )
                st.session_state.rag_chain = chain
                st.session_state.llm = llm
                st.session_state.current_docs = selected_docs
                st.session_state.chunks = all_chunks
                st.session_state.vectorstore = merged_vs

                progress_bar.empty()
                status_text.empty()
                doc_label = "、".join(selected_docs)
                st.success(f"✅ 已載入 {doc_label}（共 {len(all_chunks)} chunks）")

        # 顯示當前狀態
        if st.session_state.current_docs:
            doc_list = "、".join(st.session_state.current_docs)
            st.success(f"📄 目前文件：{doc_list}")
            if st.session_state.chunks:
                st.caption(f"共 {len(st.session_state.chunks)} 個文件片段")

    # ===== 主頁面 =====
    tab1, tab2, tab3 = st.tabs(["💬 問答", "📊 批次評估", "🔍 系統資訊"])

    with tab1:
        # 使用固定高度容器讓訊息區域可捲動，輸入框不會被推到頁面底部
        chat_container = st.container(height=600, border=False)
        with chat_container:
            # 顯示歷史對話
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg:
                        with st.expander("📄 參考來源"):
                            for src in msg["sources"]:
                                st.markdown(f"- {src.get('source', '')} 第 {src.get('page', '?')} 頁 ({src.get('type', 'text')})")
                    if "retrieved_docs" in msg and msg["retrieved_docs"]:
                        with st.expander(f"📑 檢索片段（共 {len(msg['retrieved_docs'])} 個）"):
                            for i, doc_info in enumerate(msg["retrieved_docs"]):
                                meta = doc_info["metadata"]
                                content = doc_info["content"]
                                rerank_rank = meta.get("rerank_rank", meta.get("rrf_rank", i + 1))
                                rerank_score = meta.get("rerank_score")
                                rrf = meta.get("rrf_score", 0)
                                faiss_val = meta.get("faiss_l2", -1)
                                bm25_val = meta.get("bm25_score", 0)
                                page = meta.get("page", "?")
                                doc_type = meta.get("type", "text")

                                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                                col1.metric("排名", f"#{rerank_rank}")
                                col2.metric("Rerank", f"{rerank_score:.3f}" if rerank_score is not None else "N/A")
                                col3.metric("RRF", f"{rrf:.4f}")
                                col4.metric("FAISS L2", f"{faiss_val:.2f}" if faiss_val >= 0 else "N/A")
                                col5.metric("BM25", f"{bm25_val:.2f}")

                                st.caption(f"第 {page} 頁 | {doc_type}")
                                st.text(content[:300] + ("..." if len(content) > 300 else ""))
                                st.divider()
                    if "risk" in msg and msg["risk"] != "low":
                        if msg["risk"] == "high":
                            st.error("⚠️ 高風險：此回答可能不在年報範圍內")
                        elif msg["risk"] == "medium":
                            st.warning("⚠️ 中風險：部分資訊可能需要進一步驗證")

        # 問答輸入
        if prompt := st.chat_input("輸入你的問題..."):
            if not st.session_state.rag_chain:
                st.error("請先在側邊欄上傳 PDF 並建立索引！")
            else:
                # 顯示使用者訊息
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # 生成回答
                    with st.chat_message("assistant"):
                        with st.spinner("思考中..."):
                            result = st.session_state.rag_chain.answer(
                                prompt, retrieve_k=retrieve_k, rerank_top_k=rerank_top_k
                            )

                        st.markdown(result["answer"])

                        # 來源
                        if result["sources"]:
                            with st.expander("📄 參考來源"):
                                for src in result["sources"]:
                                    st.markdown(f"- {src.get('source', '')} 第 {src.get('page', '?')} 頁 ({src.get('type', 'text')})")

                        # 風險提示
                        risk = result["hallucination_check"]["hallucination_risk"]
                        if risk == "high":
                            st.error("⚠️ 高風險：此回答可能不在年報範圍內")
                        elif risk == "medium":
                            st.warning("⚠️ 中風險：部分資訊可能需要進一步驗證")

                        # 檢索結果（chunk 內容與分數）
                        if result["retrieved_docs"]:
                            with st.expander(f"📑 檢索片段（共 {len(result['retrieved_docs'])} 個）"):
                                for i, doc_info in enumerate(result["retrieved_docs"]):
                                    meta = doc_info["metadata"]
                                    content = doc_info["content"]
                                    rerank_rank = meta.get("rerank_rank", meta.get("rrf_rank", i + 1))
                                    rerank_score = meta.get("rerank_score")
                                    rrf = meta.get("rrf_score", 0)
                                    faiss_val = meta.get("faiss_l2", -1)
                                    bm25_val = meta.get("bm25_score", 0)
                                    page = meta.get("page", "?")
                                    doc_type = meta.get("type", "text")

                                    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                                    col1.metric("排名", f"#{rerank_rank}")
                                    col2.metric("Rerank", f"{rerank_score:.3f}" if rerank_score is not None else "N/A")
                                    col3.metric("RRF", f"{rrf:.4f}")
                                    col4.metric("FAISS L2", f"{faiss_val:.2f}" if faiss_val >= 0 else "N/A")
                                    col5.metric("BM25", f"{bm25_val:.2f}")

                                    st.caption(f"第 {page} 頁 | {doc_type}")
                                    st.text(content[:300] + ("..." if len(content) > 300 else ""))
                                    st.divider()

                # 儲存到歷史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "risk": risk
                })

    with tab2:
        st.subheader("問答集批次評估")

        if not st.session_state.rag_chain:
            st.warning("請先建立索引後才能進行評估")
        else:
            qa_pairs = load_qa_pairs()
            if not qa_pairs:
                st.warning(f"找不到問答集檔案：{QA_PAIRS_PATH}")
                st.info("請將問答集 JSON 檔案放入 evaluation/ 目錄")
            else:
                st.info(f"共有 {len(qa_pairs)} 題")

                col1, col2 = st.columns(2)
                with col1:
                    start_idx = st.number_input("從第幾題開始", 1, len(qa_pairs), 1) - 1
                with col2:
                    end_idx = st.number_input("到第幾題結束", 1, len(qa_pairs), len(qa_pairs))

                selected_qa = qa_pairs[start_idx:end_idx]

                if st.button("🚀 開始評估", type="primary"):
                    evaluator = Evaluator(st.session_state.rag_chain,
                                          st.session_state.llm)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def eval_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"評估進度：{pct:.0%}")

                    report = evaluator.evaluate_all(selected_qa, eval_progress)
                    progress_bar.empty()
                    status_text.empty()

                    # 顯示結果
                    st.success("評估完成！")

                    # 指標
                    cols = st.columns(3)
                    cols[0].metric("整體正確率",
                                   f"{report['overall_accuracy']:.1%}")
                    cols[1].metric("答對題數",
                                   f"{report['correct']}/{report['total']}")
                    cols[2].metric("答錯題數",
                                   f"{report['total'] - report['correct']}")

                    # 各類別正確率
                    st.subheader("各類別正確率")
                    cat_data = []
                    for cat, stats in report["by_category"].items():
                        cat_data.append({
                            "類別": cat,
                            "正確率": f"{stats['accuracy']:.1%}",
                            "答對": stats["correct"],
                            "總題數": stats["total"]
                        })
                    if cat_data:
                        st.dataframe(pd.DataFrame(cat_data), width="stretch")

                    # 詳細結果
                    st.subheader("詳細結果")
                    detail_df = pd.DataFrame(report["details"])
                    display_cols = ["id", "category", "question",
                                    "expected_answer", "actual_answer",
                                    "is_correct", "hallucination_risk"]
                    available_cols = [c for c in display_cols if c in detail_df.columns]
                    st.dataframe(detail_df[available_cols], width="stretch")

                    # 儲存報告
                    report_path = "evaluation/results/latest_report.json"
                    os.makedirs("evaluation/results", exist_ok=True)
                    evaluator.save_report(report, report_path)
                    st.info(f"報告已儲存至 {report_path}")

    with tab3:
        st.subheader("系統架構")
        st.markdown("""
        | 元件 | 技術 |
        |------|------|
        | **Embedding** | Gemini text-embedding-004 (非對稱：doc/query 分離) |
        | **LLM** | Gemini 2.0 Flash |
        | **向量資料庫** | FAISS |
        | **Reranker** | BAAI/bge-reranker-v2-m3 (Cross-Encoder) |
        | **框架** | LangChain |
        | **前端** | Streamlit |
        | **PDF 解析** | PyMuPDF + pdfplumber |
        | **檢索策略** | FAISS 語意 + BM25 關鍵字 → RRF 融合 → Cross-Encoder Rerank |
        | **架構** | Hybrid Search → Rerank → Page Expansion → LLM → Grounding Check |
        """)

        st.subheader("目前設定")
        st.markdown(f"- **Chunk Size**: {chunk_size}")
        st.markdown(f"- **Chunk Overlap**: {chunk_overlap}")
        st.markdown(f"- **初始檢索數**: {retrieve_k}")
        st.markdown(f"- **Rerank 後保留數**: {rerank_top_k}")
        st.markdown(f"- **Reranker**: {'啟用' if ENABLE_RERANKER else '停用'}")
        st.markdown(f"- **Grounding Check**: {'啟用' if ENABLE_GROUNDING_CHECK else '停用'}")

        if st.session_state.current_docs:
            st.subheader("目前文件統計")
            if st.session_state.chunks:
                chunks = st.session_state.chunks
                st.markdown(f"- **文件片段數**: {len(chunks)}")
                text_chunks = [c for c in chunks if c.metadata.get('type') == 'text']
                table_chunks = [c for c in chunks if c.metadata.get('type') == 'table']
                st.markdown(f"- **文字片段**: {len(text_chunks)}")
                st.markdown(f"- **表格片段**: {len(table_chunks)}")
                pages = set(c.metadata.get('page', 0) for c in chunks)
                st.markdown(f"- **涵蓋頁數**: {len(pages)}")

        st.divider()
        st.subheader("金融業落地應用")
        st.markdown("""
        1. **合規審查助手** — 自動檢索年報法規遵循揭露，協助合規部門快速審查
        2. **IR 自動問答** — 股東會前自動回覆投資人問題，降低 IR 部門負擔
        3. **跨年度比較** — 多年年報 RAG 自動生成趨勢報告，輔助分析師研究
        4. **風險監控** — 結合年報自動標記風險指標變化，提升風控效率
        """)


if __name__ == "__main__":
    main()
