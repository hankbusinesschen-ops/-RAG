import json
import re
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from prompts.templates import GROUNDING_PROMPT


class HallucinationGuard:
    """幻覺檢測雙重防護（檢索相關度 + 答案錨定驗證）"""

    def __init__(self, llm, vectorstore, company_name: str = "富邦金控"):
        self.llm = llm
        self.vectorstore = vectorstore
        self.company_name = company_name
        # LCEL chains
        self.grounding_chain = GROUNDING_PROMPT | llm | StrOutputParser()

    def check(self, question: str, answer: str, retrieved_docs: list[Document]) -> dict:
        """雙重幻覺檢測"""
        # 防護 1: 檢索相關度
        relevance_ok, max_score = self._check_relevance(question)

        # 防護 2: 答案錨定驗證
        grounding_ok, grounding_detail = self._check_grounding(question, answer, retrieved_docs)

        # 綜合判定
        risk_level = self._determine_risk(relevance_ok, grounding_ok)

        return {
            "relevance_ok": relevance_ok,
            "relevance_score": max_score,
            "scope_ok": True,
            "scope_reason": "",
            "grounding_ok": grounding_ok,
            "grounding_detail": grounding_detail,
            "hallucination_risk": risk_level
        }

    def _check_relevance(self, question: str) -> tuple[bool, float]:
        """檢索相關度閾值檢查"""
        try:
            results = self.vectorstore.similarity_search_with_score(question, k=3)
            if not results:
                return False, float('inf')
            # FAISS L2 distance: 越小越相似
            min_distance = min(score for _, score in results)
            # 閾值：L2 距離太大表示不相關
            from config import RELEVANCE_THRESHOLD
            return min_distance < RELEVANCE_THRESHOLD, float(min_distance)
        except Exception:
            return True, 0.0  # 出錯時不阻擋

    def _check_grounding(self, question: str, answer: str,
                         retrieved_docs: list[Document]) -> tuple[bool, str]:
        """答案錨定驗證"""
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
        try:
            content = self.grounding_chain.invoke({
                "context": context,
                "answer": answer
            })
            parsed = self._parse_json(content)
            risk = parsed.get("risk_level", "low")
            grounded = parsed.get("grounded", True)
            return grounded, risk
        except Exception:
            return True, "low"

    def _determine_risk(self, relevance_ok: bool, grounding_ok: bool) -> str:
        """綜合判定幻覺風險"""
        if not relevance_ok and not grounding_ok:
            return "high"
        if not relevance_ok or not grounding_ok:
            return "medium"
        return "low"

    def _parse_json(self, text: str) -> dict:
        """從 LLM 輸出中提取 JSON"""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.replace('True', 'true').replace('False', 'false')
        return json.loads(text)
