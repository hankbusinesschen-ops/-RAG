import os
import json
import jieba
import numpy as np
from typing import Any
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.embeddings import get_query_embeddings


def _load_financial_terms() -> list[str]:
    """從 JSON 載入金融術語詞庫"""
    vocab_path = "config/financial_vocabulary.json"
    all_terms = []
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        for category, terms in vocab.items():
            all_terms.extend(terms)
    else:
        # fallback 基本詞彙
        all_terms = [
            "富邦金控", "富邦人壽", "富邦產險", "富邦銀行", "台北富邦銀行",
            "富邦證券", "富邦投信", "富邦華一銀行",
            "稅後淨利", "稅前淨利", "每股盈餘", "每股淨值", "股東權益",
            "資產總額", "負債總額", "淨值", "營業收入", "營業利益",
            "資本適足率", "逾放比率", "備抵覆蓋率", "流動比率",
        ]
    return list(set(all_terms))  # 去重


FINANCIAL_TERMS = _load_financial_terms()


class HybridRetriever(BaseRetriever):
    """混合檢索：FAISS 語意 + BM25 關鍵字 + 頁面圖片語意 + RRF 融合
    支援實體感知過濾"""

    vectorstore: Any  # FAISS (text)
    documents: list[Document]
    parent_docs: dict = {}
    bm25: Any = None
    query_embed_fn: Any = None
    reranker: Any = None
    image_index: Any = None  # {"vectors": np.array, "page_numbers": list}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """初始化 BM25 索引、查詢端 embedding、載入金融術語詞典"""
        super().model_post_init(__context)
        for term in FINANCIAL_TERMS:
            jieba.add_word(term)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        if self.query_embed_fn is None:
            self.query_embed_fn = get_query_embeddings()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from config import RETRIEVE_K, RERANK_TOP_K
        return self._retrieve_and_rerank(query, retrieve_k=RETRIEVE_K, rerank_top_k=RERANK_TOP_K)

    def retrieve(self, query: str, k: int = 12) -> list[Document]:
        from config import RETRIEVE_K, RERANK_TOP_K
        return self._retrieve_and_rerank(query, retrieve_k=RETRIEVE_K, rerank_top_k=RERANK_TOP_K)

    def retrieve_with_expansion(self, query: str, k: int = 12,
                                 retrieve_k: int = 20, rerank_top_k: int = 5) -> tuple[list[Document], str]:
        """檢索 chunks → rerank → 展開為完整頁面 context"""
        retrieved_docs = self._retrieve_and_rerank(query, retrieve_k=retrieve_k, rerank_top_k=rerank_top_k)

        context = self.get_parent_context(retrieved_docs)
        if not context:
            multi_source = len(set(doc.metadata.get("source", "") for doc in retrieved_docs)) > 1
            context = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', '') + ' ' if multi_source else ''}第{doc.metadata['page']}頁]\n{doc.page_content}"
                for doc in retrieved_docs
            ])

        return retrieved_docs, context

    def _retrieve_and_rerank(self, query: str, retrieve_k: int = 20, rerank_top_k: int = 5) -> list[Document]:
        """Hybrid search → 實體提升 → Rerank → top_k"""
        candidates = self._hybrid_search(query, k=retrieve_k)

        # 實體感知排序提升
        candidates = self._boost_by_entity(query, candidates)

        if self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        return candidates[:rerank_top_k]

    def _hybrid_search(self, query: str, k: int = 8) -> list[Document]:
        """FAISS 文字 + BM25 + 圖片語意，用 RRF 融合"""
        # 通道1: FAISS 文字語意搜尋
        query_vector = self.query_embed_fn.embed_query(query)
        faiss_results = self.vectorstore.similarity_search_with_score_by_vector(query_vector, k=k)

        # 通道2: BM25 關鍵字搜尋
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = bm25_scores.argsort()[-k:][::-1]
        bm25_results = [(self.documents[i], float(bm25_scores[i]))
                        for i in bm25_top_indices if bm25_scores[i] > 0]

        # 通道3: 圖片語意搜尋（如有圖片索引）
        image_results = self._image_search(query_vector, k=k) if self.image_index else []

        # RRF 多路融合
        return self._reciprocal_rank_fusion(faiss_results, bm25_results, image_results, k=k)

    def _image_search(self, query_vector, k: int = 8) -> list[tuple]:
        """用文字 query 向量搜尋圖片索引，返回匹配的頁面 Documents（支援多文件）"""
        if not self.image_index or self.image_index.get("vectors") is None:
            return []

        try:
            vectors = self.image_index["vectors"]
            page_numbers = self.image_index["page_numbers"]
            sources = self.image_index.get("sources", [None] * len(page_numbers))

            # 計算 cosine similarity
            query_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)

            # 如果維度不匹配（文字 embedding vs 圖片 embedding 可能不同維度），跳過
            if query_vec.shape[1] != vectors.shape[1]:
                return []

            # 正規化
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

            similarities = (query_norm @ vectors_norm.T).flatten()
            top_indices = similarities.argsort()[-k:][::-1]

            results = []
            for idx in top_indices:
                page_num = page_numbers[idx]
                source = sources[idx]
                score = float(similarities[idx])
                # 找到該頁的 Document（多文件時同時比對 source 和 page）
                for doc in self.documents:
                    doc_page = doc.metadata.get("page")
                    doc_source = doc.metadata.get("source")
                    if doc_page == page_num and (source is None or doc_source == source):
                        results.append((doc, score))
                        break
            return results

        except Exception:
            return []

    def _reciprocal_rank_fusion(self, faiss_results, bm25_results, image_results=None, k=8, rrf_k=60):
        """多路 RRF 融合排序"""
        doc_scores = {}
        doc_map = {}
        faiss_score_map = {}
        bm25_score_map = {}

        for rank, (doc, score) in enumerate(faiss_results):
            doc_id = doc.metadata.get('chunk_id', id(doc))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[doc_id] = doc
            faiss_score_map[doc_id] = score

        for rank, (doc, score) in enumerate(bm25_results):
            doc_id = doc.metadata.get('chunk_id', id(doc))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[doc_id] = doc
            bm25_score_map[doc_id] = score

        # 圖片通道（如有）
        if image_results:
            for rank, (doc, score) in enumerate(image_results):
                doc_id = doc.metadata.get('chunk_id', id(doc))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5 / (rrf_k + rank + 1)  # 圖片通道權重略低
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:k]

        results = []
        for rank, did in enumerate(sorted_ids):
            doc = doc_map[did]
            new_meta = dict(doc.metadata)
            new_meta['faiss_l2'] = round(faiss_score_map.get(did, -1), 4)
            new_meta['bm25_score'] = round(bm25_score_map.get(did, 0), 4)
            new_meta['rrf_score'] = round(doc_scores[did], 6)
            new_meta['rrf_rank'] = rank + 1
            results.append(Document(page_content=doc.page_content, metadata=new_meta))
        return results

    def _boost_by_entity(self, query: str, docs: list[Document]) -> list[Document]:
        """根據查詢意圖提升/降低實體相關度"""
        # 偵測查詢是否針對合併報表
        query_wants_consolidated = False
        consolidated_hints = ["金控", "合併", "整體", "總計", "集團"]
        for hint in consolidated_hints:
            if hint in query:
                query_wants_consolidated = True
                break

        # 偵測查詢是否針對特定子公司
        query_entity = None
        for doc in docs[:5]:
            entity = doc.metadata.get("entity", "")
            if entity and entity in query:
                query_entity = entity
                break

        if not query_wants_consolidated and not query_entity:
            return docs  # 無法判斷，不調整

        # 調整分數
        boosted = []
        for doc in docs:
            new_meta = dict(doc.metadata)
            rrf_score = new_meta.get("rrf_score", 0)

            entity_level = doc.metadata.get("entity_level", "")
            entity = doc.metadata.get("entity", "")

            if query_wants_consolidated and entity_level == "consolidated":
                rrf_score *= 1.5  # 提升合併報表 chunk
            elif query_wants_consolidated and entity_level == "subsidiary":
                rrf_score *= 0.7  # 降低子公司 chunk
            elif query_entity and entity == query_entity:
                rrf_score *= 1.5  # 提升目標子公司 chunk

            new_meta["rrf_score"] = rrf_score
            boosted.append(Document(page_content=doc.page_content, metadata=new_meta))

        # 重新排序
        boosted.sort(key=lambda d: d.metadata.get("rrf_score", 0), reverse=True)
        return boosted

    def _tokenize(self, text: str) -> list[str]:
        """中文分詞（含年份正規化）"""
        # 民國年份正規化：113年度 → 113年度 2024年
        import re
        text = re.sub(r'(\d{2,3})年度', lambda m: f"{m.group(0)} {int(m.group(1))+1911}年", text)
        return list(jieba.cut(text))

    def get_parent_context(self, docs: list[Document]) -> str:
        """取得 parent 文件（完整頁面內容），支援多文件"""
        page_keys = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("parent_page", doc.metadata.get("page"))
            if page:
                page_keys.add((source, page))

        # 多文件時標示來源文件名
        multi_source = len(set(k[0] for k in page_keys)) > 1 if page_keys else False
        context_parts = []
        for source, page in sorted(page_keys, key=lambda x: (x[0], x[1])):
            key = (source, page)
            if key in self.parent_docs:
                label = f"[{source} 第{page}頁]" if multi_source else f"[第{page}頁]"
                context_parts.append(f"{label}\n{self.parent_docs[key]}")

        return "\n\n---\n\n".join(context_parts)
