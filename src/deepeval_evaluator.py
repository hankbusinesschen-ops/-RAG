import os
import time
import asyncio
import statistics
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 避免 deepeval 在未設定 OPENAI_API_KEY 時印出警告
os.environ.setdefault("OPENAI_API_KEY", "not-used-gemini-only")

from deepeval.models.base_model import DeepEvalBaseLLM


class GeminiEvalModel(DeepEvalBaseLLM):
    """將 LangChain ChatGoogleGenerativeAI 包裝成 DeepEval 模型接口"""

    def __init__(self, langchain_llm):
        self._llm = langchain_llm
        super().__init__(model_name="gemini-3-flash-preview")

    def load_model(self):
        return self._llm

    def _extract_text(self, content) -> str:
        """從 LangChain response.content 提取純文字。
        Gemini 3 Flash 回傳 list[dict] 而非 str。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def generate(self, prompt, **kwargs):
        from langchain_core.messages import HumanMessage
        response = self._llm.invoke([HumanMessage(content=str(prompt))])
        return self._extract_text(response.content)

    async def a_generate(self, prompt, **kwargs):
        # 用同步 invoke 取代 ainvoke，避免 LangChain async 在手動 event loop 中產生 orphan tasks
        from langchain_core.messages import HumanMessage
        response = self._llm.invoke([HumanMessage(content=str(prompt))])
        return self._extract_text(response.content)

    def get_model_name(self) -> str:
        return "gemini-3-flash-preview"


@dataclass
class EvaluationItem:
    """評估題目的輸入結構"""
    question: str
    expected_answer: Optional[str] = None   # ContextualRecall / GEval_Correctness 必填
    context_hint: Optional[str] = None      # 附加於問題前，影響 retrieval


@dataclass
class SingleRunResult:
    """單次執行結果"""
    run_index: int
    actual_answer: str
    retrieval_context: list
    hallucination_risk: str
    metric_scores: dict
    metric_reasons: dict


@dataclass
class ItemResult:
    """單題完整評估結果"""
    item: EvaluationItem
    runs: list
    mean_scores: dict
    std_scores: dict
    consistency_score: float


# 需要 expected_output 的 metric
NEEDS_EXPECTED = {"ContextualRecall", "GEval_Correctness"}
# 需要 retrieval_context 的 metric
NEEDS_CONTEXT = {"Faithfulness", "ContextualRelevancy", "ContextualRecall"}


class DeepEvalEvaluator:
    """DeepEval 整合評估器：支援動態題目、多次重複、多種指標"""

    def __init__(self, rag_chain, llm):
        self.chain = rag_chain
        self.eval_model = GeminiEvalModel(llm)

    def _measure_in_thread(self, metric, test_case) -> tuple:
        """在獨立 thread（含獨立 event loop）中執行 metric.measure()，
        完全隔離於 Streamlit 的 event loop 之外。"""
        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metric.measure(test_case)
                return metric.score, getattr(metric, "reason", "") or ""
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            return future.result(timeout=120)

    def _build_metrics(self, selected: list) -> dict:
        """依勾選名稱建立 Metric 實例（延遲 import）"""
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRelevancyMetric,
            ContextualRecallMetric,
            GEval,
        )
        from deepeval.test_case import LLMTestCaseParams

        metrics = {}
        if "AnswerRelevancy" in selected:
            metrics["AnswerRelevancy"] = AnswerRelevancyMetric(
                threshold=0.5, model=self.eval_model, include_reason=True
            )
        if "Faithfulness" in selected:
            metrics["Faithfulness"] = FaithfulnessMetric(
                threshold=0.5, model=self.eval_model, include_reason=True
            )
        if "ContextualRelevancy" in selected:
            metrics["ContextualRelevancy"] = ContextualRelevancyMetric(
                threshold=0.5, model=self.eval_model, include_reason=True
            )
        if "ContextualRecall" in selected:
            metrics["ContextualRecall"] = ContextualRecallMetric(
                threshold=0.5, model=self.eval_model, include_reason=True
            )
        if "GEval_Correctness" in selected:
            metrics["GEval_Correctness"] = GEval(
                name="Correctness",
                criteria=(
                    "判斷 actual output 是否正確回答了問題，並與 expected output 語意一致。"
                    "數字型需數值相符（允許格式差異）；文字型需涵蓋核心資訊；"
                    "幻覺檢測型需確認系統是否明確拒答超範圍問題。"
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                model=self.eval_model,
                threshold=0.5,
            )
        return metrics

    def _run_single(self, item: EvaluationItem, metrics: dict, run_idx: int) -> SingleRunResult:
        """執行一次問答並評估所有選定指標"""
        from deepeval.test_case import LLMTestCase

        # 組合問題（加 context_hint 前綴）
        question_with_hint = item.question
        if item.context_hint:
            question_with_hint = f"[背景提示：{item.context_hint}]\n{item.question}"

        # 暫時停用 grounding check 加速（同 evaluator.py 模式）
        original_guard = self.chain.guard
        self.chain.guard = None
        try:
            result = self.chain.answer(question_with_hint)
        finally:
            self.chain.guard = original_guard

        actual_answer = result["answer"]
        retrieval_context = [d["content"] for d in result.get("retrieved_docs", [])]
        hallucination_risk = result["hallucination_check"]["hallucination_risk"]

        # 建立 DeepEval 測試案例（input 使用原始問題，不含 hint）
        test_case = LLMTestCase(
            input=item.question,
            actual_output=actual_answer,
            expected_output=item.expected_answer or "",
            retrieval_context=retrieval_context,
        )

        # 逐一評估 metric（在獨立 thread 中執行，避免與 Streamlit event loop 衝突）
        metric_scores = {}
        metric_reasons = {}
        for name, metric in metrics.items():
            if name in NEEDS_EXPECTED and not item.expected_answer:
                metric_scores[name] = None
                metric_reasons[name] = "未提供預期答案，跳過"
                continue
            if name in NEEDS_CONTEXT and not retrieval_context:
                metric_scores[name] = None
                metric_reasons[name] = "無檢索內容，跳過"
                continue
            try:
                score, reason = self._measure_in_thread(metric, test_case)
                metric_scores[name] = score
                metric_reasons[name] = reason
            except Exception as e:
                metric_scores[name] = None
                metric_reasons[name] = f"評估失敗：{str(e)[:120]}"

        return SingleRunResult(
            run_index=run_idx,
            actual_answer=actual_answer,
            retrieval_context=retrieval_context,
            hallucination_risk=hallucination_risk,
            metric_scores=metric_scores,
            metric_reasons=metric_reasons,
        )

    def _calc_consistency(self, runs: list, selected_metrics: list) -> float:
        """多次回答一致性（以 metric 得分標準差衡量）"""
        if len(runs) <= 1:
            return 1.0
        all_stds = []
        for m_name in selected_metrics:
            vals = [r.metric_scores.get(m_name) for r in runs
                    if r.metric_scores.get(m_name) is not None]
            if len(vals) > 1:
                all_stds.append(statistics.stdev(vals))
        avg_std = sum(all_stds) / len(all_stds) if all_stds else 0.0
        return max(0.0, 1.0 - avg_std)

    def run_evaluation(
        self,
        qa_items: list,
        selected_metrics: list,
        n_repeats: int = 1,
        progress_callback=None,
    ) -> list:
        """完整評估流程，回傳 list[ItemResult]"""
        metrics = self._build_metrics(selected_metrics)
        results = []
        total_calls = len(qa_items) * n_repeats

        for i, item in enumerate(qa_items):
            runs = []
            for r in range(n_repeats):
                call_idx = i * n_repeats + r
                run = self._run_single(item, metrics, r)
                runs.append(run)
                if progress_callback:
                    progress_callback((call_idx + 1) / total_calls)
                if call_idx < total_calls - 1:
                    time.sleep(2)

            # 聚合統計
            mean_scores = {}
            std_scores = {}
            for m_name in selected_metrics:
                vals = [rn.metric_scores.get(m_name) for rn in runs
                        if rn.metric_scores.get(m_name) is not None]
                mean_scores[m_name] = float(np.mean(vals)) if vals else None
                std_scores[m_name] = float(np.std(vals)) if len(vals) > 1 else 0.0

            consistency = self._calc_consistency(runs, selected_metrics)
            results.append(ItemResult(
                item=item,
                runs=runs,
                mean_scores=mean_scores,
                std_scores=std_scores,
                consistency_score=consistency,
            ))

        return results

    def to_export_dict(self, results: list) -> list:
        """序列化為可 JSON dump 的格式"""
        output = []
        for r in results:
            output.append({
                "question": r.item.question,
                "expected_answer": r.item.expected_answer,
                "context_hint": r.item.context_hint,
                "mean_scores": r.mean_scores,
                "std_scores": r.std_scores,
                "consistency_score": r.consistency_score,
                "runs": [
                    {
                        "run_index": run.run_index,
                        "actual_answer": run.actual_answer,
                        "hallucination_risk": run.hallucination_risk,
                        "metric_scores": run.metric_scores,
                        "metric_reasons": run.metric_reasons,
                    }
                    for run in r.runs
                ],
            })
        return output
