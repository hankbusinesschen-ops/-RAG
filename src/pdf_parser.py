import os
import re
import json
import hashlib
import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
from config import (
    GOOGLE_API_KEY, ENABLE_VISION_FALLBACK, VISION_MODEL,
    VISION_CACHE_DIR, MIN_CHINESE_CHARS
)


class PDFParser:
    """解析 PDF，提取文字與表格，保留頁碼 metadata。
    支援旋轉頁面偵測、Gemini Vision fallback、頁首頁尾移除。
    """

    def __init__(self):
        self._header_footer_patterns = None  # 延遲初始化

    def parse(self, pdf_path: str) -> list[Document]:
        """主解析流程：文字 + 表格"""
        # Step 1: 先掃描全文件以偵測 header/footer 模式
        self._detect_header_footer_patterns(pdf_path)

        # Step 2: 提取文字（含旋轉頁面偵測與 Vision fallback）
        text_docs = self._extract_text_with_fitz(pdf_path)

        # Step 3: 提取表格
        table_docs = self._extract_tables_with_pdfplumber(pdf_path)

        # Step 4: 合併 & 去重
        documents = self._merge_and_deduplicate(text_docs, table_docs)

        # Step 5: 渲染頁面圖片（供圖片 embedding 使用）
        self._render_page_images(pdf_path)

        return documents

    def _extract_text_with_fitz(self, pdf_path: str) -> list[Document]:
        """PyMuPDF 提取文字，含旋轉偵測與 Vision fallback"""
        doc = fitz.open(pdf_path)
        documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = self._clean_text(text)

            # 品質檢查：旋轉頁面或中文字元過少
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            is_rotated = page.rotation != 0
            has_visual_content = self._page_has_content(page)
            needs_vision = is_rotated or (chinese_chars < MIN_CHINESE_CHARS and has_visual_content)

            if needs_vision and ENABLE_VISION_FALLBACK:
                vision_text = self._extract_with_vision(page, page_num, pdf_path)
                if vision_text:
                    text = vision_text

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": os.path.basename(pdf_path),
                        "type": "text",
                        "extraction_method": "vision" if (needs_vision and ENABLE_VISION_FALLBACK) else "pymupdf",
                        "is_rotated": is_rotated,
                    }
                ))
        doc.close()
        return documents

    def _page_has_content(self, page) -> bool:
        """檢查頁面是否有視覺內容（即使文字提取失敗）"""
        # 檢查是否有文字區塊（即使無法正確排序）
        blocks = page.get_text("blocks")
        if len(blocks) > 3:
            return True
        # 檢查是否有圖片
        images = page.get_images()
        if images:
            return True
        # 檢查頁面是否有繪製內容
        drawings = page.get_drawings()
        if len(drawings) > 5:
            return True
        return False

    def _extract_with_vision(self, page, page_num: int, pdf_path: str) -> str:
        """使用 Gemini Vision API 提取頁面內容，含快取機制"""
        # 生成快取 key
        cache_key = self._get_vision_cache_key(pdf_path, page_num)
        cache_path = os.path.join(VISION_CACHE_DIR, f"{cache_key}.txt")

        # 檢查快取
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        try:
            from google import genai
            from google.genai import types
            from prompts.templates import VISION_EXTRACTION_PROMPT

            # 渲染頁面為圖片
            pixmap = page.get_pixmap(dpi=300)
            img_bytes = pixmap.tobytes("png")

            # 呼叫 Gemini Vision API
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    types.Part.from_text(text=VISION_EXTRACTION_PROMPT),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ]
            )

            extracted_text = response.text.strip() if response.text else ""

            # 寫入快取
            if extracted_text:
                os.makedirs(VISION_CACHE_DIR, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

            return extracted_text

        except Exception as e:
            print(f"[Vision] 第{page_num + 1}頁 Vision 提取失敗: {e}")
            return ""

    def _get_vision_cache_key(self, pdf_path: str, page_num: int) -> str:
        """生成 Vision 快取 key（基於檔案+頁碼的 hash）"""
        file_stat = os.stat(pdf_path)
        key_str = f"{os.path.basename(pdf_path)}_{file_stat.st_size}_{page_num}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _detect_header_footer_patterns(self, pdf_path: str):
        """掃描全文件，統計出現頻率高的頁首/頁尾行"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages < 5:
            self._header_footer_patterns = set()
            doc.close()
            return

        line_counts = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 取前3行和後3行
            candidates = lines[:3] + lines[-3:] if len(lines) > 6 else lines
            for line in candidates:
                # 忽略太長的行（可能是正文）
                if len(line) > 50:
                    continue
                line_counts[line] = line_counts.get(line, 0) + 1

        doc.close()

        # 出現在 >40% 頁面的短行視為 header/footer
        threshold = total_pages * 0.4
        self._header_footer_patterns = {
            line for line, count in line_counts.items() if count > threshold
        }

    def _clean_text(self, text: str) -> str:
        """中文 PDF 文字清理（通用化，不硬編碼公司名稱）"""
        # 修復中文斷行
        text = re.sub(r'(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])', '', text)
        # 移除多餘空白但保留段落分隔
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 全形數字轉半形
        text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

        # 移除頁碼行（如 "- 1 -" 或 "1"）
        text = re.sub(r'^-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)

        # 移除統計偵測到的 header/footer
        if self._header_footer_patterns:
            lines = text.split("\n")
            cleaned_lines = [
                l for l in lines
                if l.strip() not in self._header_footer_patterns
            ]
            text = "\n".join(cleaned_lines)

        return text.strip()

    def _extract_tables_with_pdfplumber(self, pdf_path: str) -> list[Document]:
        """pdfplumber 提取表格，轉為 Markdown 格式"""
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    md_table = self._table_to_markdown(table)
                    if md_table.strip():
                        title = self._extract_table_title(page, table)
                        content = f"[表格: {title}]\n{md_table}" if title else f"[表格]\n{md_table}"
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "page": page_num,
                                "source": os.path.basename(pdf_path),
                                "type": "table",
                                "table_index": i
                            }
                        ))

        # 跨頁表格合併
        documents = self._merge_cross_page_tables(documents)
        return documents

    def _merge_cross_page_tables(self, table_docs: list[Document]) -> list[Document]:
        """偵測並合併跨頁延續的表格"""
        if len(table_docs) < 2:
            return table_docs

        merged = []
        skip_next = set()

        for i, doc in enumerate(table_docs):
            if i in skip_next:
                continue

            # 嘗試與下一頁的表格合併
            current = doc
            j = i + 1
            while j < len(table_docs):
                next_doc = table_docs[j]
                next_page = next_doc.metadata["page"]
                curr_page = current.metadata["page"]

                # 只合併相鄰頁的表格
                if next_page != curr_page + 1:
                    break

                # 檢查是否為延續表格
                if self._is_continuation_table(current, next_doc):
                    # 合併：將下一個表格的資料行附加到當前表格
                    current = self._merge_two_tables(current, next_doc)
                    skip_next.add(j)
                    j += 1
                else:
                    break

            merged.append(current)

        return merged

    def _is_continuation_table(self, table1: Document, table2: Document) -> bool:
        """判斷 table2 是否為 table1 的延續"""
        content1 = table1.page_content
        content2 = table2.page_content

        # 檢查續頁標記
        continuation_markers = ["（續）", "(續)", "(continued)", "承上頁", "續上表"]
        for marker in continuation_markers:
            if marker in content2[:100]:
                return True

        # 比較欄數是否一致
        lines1 = [l for l in content1.split("\n") if l.startswith("|")]
        lines2 = [l for l in content2.split("\n") if l.startswith("|")]
        if lines1 and lines2:
            cols1 = len(lines1[0].split("|"))
            cols2 = len(lines2[0].split("|"))
            # 欄數相同且 table2 沒有分隔線（表示無新表頭）
            has_separator = any("---" in l for l in lines2[:3])
            if cols1 == cols2 and not has_separator:
                return True

        return False

    def _merge_two_tables(self, table1: Document, table2: Document) -> Document:
        """合併兩個表格 Document"""
        # 從 table2 中提取資料行（跳過表頭和分隔線）
        lines2 = table2.page_content.split("\n")
        data_lines = []
        skip_header = True
        for line in lines2:
            if skip_header:
                if line.startswith("|") and "---" in line:
                    skip_header = False
                    continue
                if line.startswith("|"):
                    continue
                if line.startswith("[表格"):
                    continue
                skip_header = False
            if line.startswith("|"):
                data_lines.append(line)

        merged_content = table1.page_content
        if data_lines:
            merged_content += "\n" + "\n".join(data_lines)

        return Document(
            page_content=merged_content,
            metadata={
                **table1.metadata,
                "page_end": table2.metadata["page"],
                "merged_pages": True,
            }
        )

    def _merge_and_deduplicate(self, text_docs: list[Document], table_docs: list[Document]) -> list[Document]:
        """合併文字與表格文件，按頁碼排序"""
        all_docs = text_docs + table_docs
        all_docs.sort(key=lambda d: (d.metadata["page"], 0 if d.metadata["type"] == "text" else 1))
        return all_docs

    def _table_to_markdown(self, table: list[list]) -> str:
        """表格轉 Markdown，保留所有數字精度"""
        if not table or not table[0]:
            return ""
        clean_table = []
        for row in table:
            clean_row = []
            for cell in row:
                cell_str = str(cell).strip() if cell is not None else ""
                # 清理換行
                cell_str = cell_str.replace("\n", " ")
                clean_row.append(cell_str)
            clean_table.append(clean_row)

        # 確保所有行長度一致
        max_cols = max(len(row) for row in clean_table)
        clean_table = [row + [""] * (max_cols - len(row)) for row in clean_table]

        header = "| " + " | ".join(clean_table[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in clean_table[1:]]
        return "\n".join([header, separator] + rows)

    def _extract_table_title(self, page, table) -> str:
        """嘗試提取表格上方的標題文字"""
        try:
            text = page.extract_text() or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 尋找可能的表格標題（含「表」字或較短的行）
            for line in lines[:10]:
                if ("表" in line or "Table" in line) and len(line) < 60:
                    return line
            return ""
        except Exception:
            return ""

    def _render_page_images(self, pdf_path: str):
        """為每頁渲染 PNG 縮圖（供圖片 embedding 使用）"""
        from config import PAGE_IMAGES_DIR
        os.makedirs(PAGE_IMAGES_DIR, exist_ok=True)

        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        img_dir = os.path.join(PAGE_IMAGES_DIR, basename)
        os.makedirs(img_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
            if not os.path.exists(img_path):
                page = doc[page_num]
                pixmap = page.get_pixmap(dpi=150)
                pixmap.save(img_path)
        doc.close()
