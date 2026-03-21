import json
import re
from langchain_core.output_parsers import StrOutputParser
from prompts.templates import QUERY_ANALYSIS_PROMPT


class QueryProcessor:
    """查詢預處理：拆解複合問題、生成多角度查詢"""

    def __init__(self, llm):
        self.llm = llm
        # LCEL chain: prompt | llm | parser
        self.analysis_chain = QUERY_ANALYSIS_PROMPT | llm | StrOutputParser()

    def process(self, question: str) -> dict:
        """分析問題並返回處理策略"""
        content = self.analysis_chain.invoke({"question": question})

        try:
            parsed = self._parse_json(content)
        except Exception:
            # fallback: 當 LLM 回傳無法解析時，使用預設值
            parsed = {
                "is_compound": False,
                "needs_calculation": False,
                "sub_questions": [question],
                "search_queries": [question],
                "expected_answer_type": "文字"
            }

        # 確保 sub_questions 至少包含原始問題
        if not parsed.get("sub_questions"):
            parsed["sub_questions"] = [question]
        if not parsed.get("search_queries"):
            parsed["search_queries"] = [question]

        return parsed

    def _parse_json(self, text: str) -> dict:
        """從 LLM 輸出中提取 JSON"""
        text = text.strip()
        # 移除可能的 markdown 包裹
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        # 修復常見 JSON 錯誤：true/false
        text = text.replace('True', 'true').replace('False', 'false')
        return json.loads(text)
