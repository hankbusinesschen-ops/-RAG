import json
import re
import time
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from src.chain import RAGChain
from prompts.templates import JUDGE_PROMPT


class Evaluator:
    """自動化評估模組：跑完 30 題並精確計算 Accuracy"""

    def __init__(self, rag_chain: RAGChain, llm):
        self.chain = rag_chain
        self.llm = llm
        # LCEL chain: prompt | llm | parser
        self.judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

    def evaluate_all(self, qa_pairs: list[dict],
                     progress_callback=None) -> dict:
        """跑完所有題目並計算正確率"""
        results = []
        total = len(qa_pairs)

        for i, qa in enumerate(qa_pairs):
            result = self._evaluate_single(qa)
            results.append(result)
            if progress_callback:
                progress_callback((i + 1) / total)
            # 避免 Gemini API rate limit（每題間隔 2 秒）
            if i < total - 1:
                time.sleep(2)

        report = self._generate_report(results)
        return report

    def _evaluate_single(self, qa: dict) -> dict:
        """評估單題（停用 grounding check 以加速並避免 rate limit）"""
        # 暫時停用 grounding check
        original_guard = self.chain.guard
        self.chain.guard = None
        try:
            response = self.chain.answer(qa["question"])
        finally:
            self.chain.guard = original_guard

        # 用 LLM 判定回答是否正確
        is_correct = self._judge_answer(
            question=qa["question"],
            expected=qa["answer"],
            actual=response["answer"]
        )

        return {
            "id": qa["id"],
            "category": qa.get("category", ""),
            "type": qa.get("type", ""),
            "question": qa["question"],
            "expected_answer": qa["answer"],
            "actual_answer": response["answer"],
            "is_correct": is_correct,
            "hallucination_risk": response["hallucination_check"]["hallucination_risk"],
            "source_pages": [s.get("page") for s in response["sources"]]
        }

    def _judge_answer(self, question: str, expected: str, actual: str) -> bool:
        """用 LLM 判定回答是否等價於標準答案"""
        try:
            content = self.judge_chain.invoke({"expected": expected, "actual": actual})
            parsed = self._parse_json(content)
            return parsed.get("is_correct", False)
        except Exception:
            return False

    def _generate_report(self, results: list[dict]) -> dict:
        """生成評估報告"""
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])

        # 按類別統計
        categories = {}
        for r in results:
            cat = r.get("category", "未分類")
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}
            categories[cat]["total"] += 1
            if r["is_correct"]:
                categories[cat]["correct"] += 1

        return {
            "overall_accuracy": correct / total if total > 0 else 0,
            "total": total,
            "correct": correct,
            "by_category": {
                cat: {
                    "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                    "correct": v["correct"],
                    "total": v["total"]
                }
                for cat, v in categories.items()
            },
            "details": results
        }

    def save_report(self, report: dict, output_path: str):
        """儲存評估報告"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 同時產生 CSV
        csv_path = output_path.replace(".json", ".csv")
        df = pd.DataFrame(report["details"])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _parse_json(self, text: str) -> dict:
        """從 LLM 輸出中提取 JSON"""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.replace('True', 'true').replace('False', 'false')
        return json.loads(text)
