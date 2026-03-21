import os
import re
import json
from langchain_core.documents import Document


class MetadataEnricher:
    """為 chunk 添加實體感知 metadata（支援多家金控）"""

    def __init__(self, entities_path: str = "config/entities.json"):
        self.entities_config = {}
        self.company_name = None
        self.company_config = None

        if os.path.exists(entities_path):
            with open(entities_path, "r", encoding="utf-8") as f:
                self.entities_config = json.load(f)

    def detect_company(self, documents: list[Document]) -> str:
        """從文件前幾頁自動偵測屬於哪家金控"""
        # 取前 10 頁的文字
        early_text = ""
        for doc in documents:
            if doc.metadata.get("page", 999) <= 10:
                early_text += doc.page_content + "\n"

        # 計算每家金控名稱出現次數
        best_company = None
        best_count = 0

        for company, config in self.entities_config.items():
            count = 0
            for keyword in config.get("consolidated", []):
                count += early_text.count(keyword)
            if count > best_count:
                best_count = count
                best_company = company

        if best_company:
            self.company_name = best_company
            self.company_config = self.entities_config[best_company]

        return best_company or "unknown"

    def enrich_documents(self, documents: list[Document], toc_parser=None) -> list[Document]:
        """為所有 chunk 添加實體 metadata"""
        if not self.company_config:
            return documents

        # 建立章節→實體映射（基於 TOC 的子公司章節）
        section_entity_map = self._build_section_entity_map(toc_parser)

        for doc in documents:
            entity, entity_level = self._detect_entity(doc, section_entity_map)
            doc.metadata["company"] = self.company_name
            doc.metadata["entity"] = entity
            doc.metadata["entity_level"] = entity_level

        return documents

    def _build_section_entity_map(self, toc_parser) -> dict:
        """從 TOC 章節名稱中提取子公司對應"""
        mapping = {}
        if not toc_parser or not toc_parser.sections:
            return mapping

        subsidiaries = self.company_config.get("subsidiaries", {})

        for section in toc_parser.sections:
            title = section.title
            for sub_name, keywords in subsidiaries.items():
                if any(kw in title for kw in keywords) or sub_name in title:
                    # 這個章節及其所有子頁面對應到此子公司
                    for page in range(section.physical_page, section.end_page + 1):
                        mapping[page] = sub_name
                    break

        return mapping

    def _detect_entity(self, doc: Document, section_entity_map: dict) -> tuple[str, str]:
        """偵測 chunk 的實體歸屬。
        優先順序：TOC 章節映射 > 關鍵字計數（僅在子公司章節內使用）
        不在子公司章節的頁面一律視為 consolidated。
        """
        page = doc.metadata.get("page", 0)

        # 最高優先：TOC 章節映射（明確標記為某子公司的章節）
        if page in section_entity_map:
            return section_entity_map[page], "subsidiary"

        # 不在任何子公司章節的頁面 → 視為合併報表層級
        # 這包括：致股東報告書、公司治理報告、募資情形、財務狀況等
        return self.company_name, "consolidated"
