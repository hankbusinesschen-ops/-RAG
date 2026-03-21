import json
import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一個文件相關性排序助手。給定一個問題和多個候選段落，請根據與問題的相關性對段落排序。

回傳 JSON 格式（不要加 markdown 標記）：
{{"ranking": [段落編號1, 段落編號2, ...]}}

只回傳最相關的前 {top_k} 個段落編號，從最相關到最不相關排列。"""),
    ("human", "問題：{query}\n\n候選段落：\n{passages}")
])


class LLMReranker:
    """用 Gemini LLM 做 reranking，零本地記憶體"""

    def __init__(self, llm):
        self.chain = RERANK_PROMPT | llm | StrOutputParser()

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        if not docs or top_k <= 0:
            return []

        # 將候選段落編號列出（截取前 300 字避免 token 過多）
        passages_text = ""
        for i, doc in enumerate(docs):
            content = doc.page_content[:300]
            page = doc.metadata.get("page", "?")
            passages_text += f"\n[{i}] (第{page}頁) {content}\n"

        try:
            result = self.chain.invoke({
                "query": query,
                "passages": passages_text,
                "top_k": top_k,
            })
            ranking = self._parse_ranking(result, len(docs), top_k)
        except Exception:
            # 失敗時退回原始順序
            ranking = list(range(min(top_k, len(docs))))

        results = []
        for rank, idx in enumerate(ranking):
            if idx < len(docs):
                doc = docs[idx]
                new_meta = dict(doc.metadata)
                new_meta['rerank_score'] = round(1.0 - rank / len(ranking), 4)
                new_meta['rerank_rank'] = rank + 1
                results.append(Document(page_content=doc.page_content, metadata=new_meta))

        return results

    def _parse_ranking(self, text: str, total: int, top_k: int) -> list[int]:
        """從 LLM 輸出中解析排序結果"""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            parsed = json.loads(text)
            ranking = parsed.get("ranking", [])
        except (json.JSONDecodeError, AttributeError):
            # 嘗試直接提取數字列表
            numbers = re.findall(r'\d+', text)
            ranking = [int(n) for n in numbers]

        # 過濾有效的索引，去重
        seen = set()
        valid = []
        for idx in ranking:
            if isinstance(idx, int) and 0 <= idx < total and idx not in seen:
                seen.add(idx)
                valid.append(idx)
            if len(valid) >= top_k:
                break

        return valid
