import json
import os
from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from src.retriever import HybridRetriever
from src.hallucination import HallucinationGuard
from prompts.templates import UNIFIED_QA_PROMPT
from config import RELEVANCE_THRESHOLD, ENABLE_GROUNDING_CHECK, RETRIEVE_K, RERANK_TOP_K

# 動態載入競爭公司（從 entities.json 中排除當前公司）
PREDICTION_KEYWORDS = ["預測", "預估", "預計將", "未來", "明年", "後年"]


def _load_rival_companies(current_company: str = None) -> list[str]:
    """從 entities.json 載入其他金控名稱（排除當前公司）。"""
    return _load_rival_companies_multi({current_company} if current_company else set())


def _load_rival_companies_multi(exclude_companies: set[str] = None) -> list[str]:
    """從 entities.json 載入競爭公司名稱，排除所有已選中的公司。
    只加入公司專有名詞，不加入通用詞彙（如「合併」、「人壽」）。
    """
    if exclude_companies is None:
        exclude_companies = set()
    entities_path = "config/entities.json"
    rivals = []
    generic_words = {"合併", "金控合併", "金控整體", "人壽", "銀行", "證券", "產險",
                     "人壽公司", "壽險子公司", "證券子公司", "產險子公司", "投信子公司",
                     "創投子公司", "保代子公司", "保險代理", "資產管理", "香港子公司"}
    if os.path.exists(entities_path):
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        for company, config in entities.items():
            if company in exclude_companies:
                continue
            for kw in config.get("consolidated", []):
                if kw not in generic_words:
                    rivals.append(kw)
            for sub_name, keywords in config.get("subsidiaries", {}).items():
                rivals.append(sub_name)
                for kw in keywords:
                    if kw not in generic_words:
                        rivals.append(kw)
    return list(set(rivals))


class RAGChain:
    """RAG Chain：Hybrid Search + Rerank + Entity Awareness + Grounding Check"""

    def __init__(self, llm, retriever: HybridRetriever, vectorstore=None,
                 current_company: str = None, current_companies: list[str] = None):
        self.llm = llm
        self.retriever = retriever
        self.current_company = current_company

        # 動態載入競爭公司列表（排除所有已選公司）
        companies_to_exclude = set()
        if current_company:
            companies_to_exclude.add(current_company)
        if current_companies:
            companies_to_exclude.update(c for c in current_companies if c)
        self.rival_companies = _load_rival_companies_multi(companies_to_exclude)

        # 單一 LCEL chain
        self.qa_chain = UNIFIED_QA_PROMPT | llm | StrOutputParser()

        # Grounding Check
        self.guard = None
        if ENABLE_GROUNDING_CHECK and vectorstore:
            try:
                self.guard = HallucinationGuard(llm, vectorstore)
            except Exception:
                pass

    def answer(self, question: str, retrieve_k: int = None,
               rerank_top_k: int = None) -> dict:
        """完整問答流程"""
        if retrieve_k is None:
            retrieve_k = RETRIEVE_K
        if rerank_top_k is None:
            rerank_top_k = RERANK_TOP_K

        # Step 1: Scope check
        scope_result = self._check_scope(question)
        if scope_result:
            return self._build_refusal(question, scope_result)

        # Step 2: 檢索 + rerank + 頁面展開
        retrieved_docs, context = self.retriever.retrieve_with_expansion(
            question, retrieve_k=retrieve_k, rerank_top_k=rerank_top_k
        )

        # Step 3: Risk assessment
        risk_level = self._assess_risk(retrieved_docs)

        # Step 4: LLM 回答
        if risk_level == "high":
            answer_text = "根據所提供的年報內容，無法找到此資訊。"
        else:
            answer_text = self.qa_chain.invoke({
                "context": context,
                "question": question
            })

        # Step 5: Grounding Check
        grounding_ok = True
        grounding_detail = ""
        if self.guard and risk_level != "high":
            try:
                from langchain_core.documents import Document
                guard_docs = [Document(page_content=d.page_content, metadata=d.metadata)
                              for d in retrieved_docs]
                guard_result = self.guard.check(question, answer_text, guard_docs)
                grounding_ok = guard_result.get("grounding_ok", True)
                grounding_detail = guard_result.get("grounding_detail", "")
                if not grounding_ok and risk_level == "low":
                    risk_level = "medium"
            except Exception:
                pass

        return {
            "question": question,
            "answer": answer_text,
            "sources": [doc.metadata for doc in retrieved_docs],
            "retrieved_docs": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in retrieved_docs
            ],
            "query_analysis": {},
            "hallucination_check": {
                "relevance_ok": risk_level != "high",
                "relevance_score": self._get_min_faiss_score(retrieved_docs),
                "scope_ok": True,
                "scope_reason": "",
                "grounding_ok": grounding_ok,
                "grounding_detail": grounding_detail,
                "hallucination_risk": risk_level,
            }
        }

    def _check_scope(self, question: str) -> Optional[str]:
        """Scope check：其他公司 / 預測未來"""
        for company in self.rival_companies:
            if company in question:
                return f"此為{company}相關問題，超出年報範圍"
        for kw in PREDICTION_KEYWORDS:
            if kw in question:
                return "年報為歷史資料，無法提供未來預測"
        return None

    def _build_refusal(self, question: str, reason: str) -> dict:
        return {
            "question": question,
            "answer": f"根據所提供的年報內容，無法回答此問題。原因：{reason}",
            "sources": [],
            "retrieved_docs": [],
            "query_analysis": {},
            "hallucination_check": {
                "relevance_ok": False,
                "relevance_score": float('inf'),
                "scope_ok": False,
                "scope_reason": reason,
                "grounding_ok": False,
                "grounding_detail": "",
                "hallucination_risk": "high",
            }
        }

    def _assess_risk(self, retrieved_docs: list) -> str:
        min_score = self._get_min_faiss_score(retrieved_docs)
        if min_score == float('inf'):
            return "high"
        if min_score > RELEVANCE_THRESHOLD:
            return "high"
        return "low"

    def _get_min_faiss_score(self, retrieved_docs: list) -> float:
        scores = []
        for doc in retrieved_docs:
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            faiss_l2 = meta.get('faiss_l2', -1)
            if faiss_l2 >= 0:
                scores.append(faiss_l2)
        return min(scores) if scores else float('inf')
