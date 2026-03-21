import re
import fitz
from dataclasses import dataclass, field


@dataclass
class Section:
    """目錄章節"""
    section_id: str       # e.g., "2.1.1"
    title: str            # e.g., "董事資料"
    doc_page: int         # 文件內頁碼（目錄標示的頁碼）
    physical_page: int    # 實體頁碼（PDF 0-indexed + 1）
    level: int            # 層級深度（1=頂層, 2=次層...）
    children: list = field(default_factory=list)
    end_page: int = 0     # 章節結束頁（由後處理計算）


class TOCParser:
    """解析金控年報目錄，建立章節樹和頁面映射。
    通用化設計：自動偵測目錄頁、計算頁碼偏移。
    """

    def __init__(self):
        self.sections: list[Section] = []
        self.page_offset: int = 0  # physical_page = doc_page + offset
        self.page_to_section: dict[int, Section] = {}

    def parse(self, pdf_path: str) -> list[Section]:
        """解析 PDF 目錄，返回章節列表"""
        doc = fitz.open(pdf_path)

        # Step 1: 找到目錄頁
        toc_text = self._find_toc_pages(doc)
        if not toc_text:
            doc.close()
            return []

        # Step 2: 解析目錄行
        raw_sections = self._parse_toc_lines(toc_text)
        if not raw_sections:
            doc.close()
            return []

        # Step 3: 計算頁碼偏移（文件頁碼 vs 實體頁碼）
        self.page_offset = self._detect_page_offset(doc, raw_sections)

        # Step 4: 構建章節列表（加入實體頁碼）
        self.sections = []
        for sec_id, title, doc_page, level in raw_sections:
            physical_page = doc_page + self.page_offset
            self.sections.append(Section(
                section_id=sec_id,
                title=title,
                doc_page=doc_page,
                physical_page=physical_page,
                level=level,
            ))

        # Step 5: 計算每個章節的結束頁
        self._compute_end_pages(len(doc))

        # Step 6: 建立 page → section 映射
        self._build_page_mapping()

        doc.close()
        return self.sections

    def get_section_for_page(self, physical_page: int) -> dict:
        """取得指定頁面的章節資訊"""
        section = self.page_to_section.get(physical_page)
        if not section:
            return {}
        return {
            "section_id": section.section_id,
            "section_title": section.title,
            "section_path": self._get_section_path(section),
        }

    def _get_section_path(self, section: Section) -> str:
        """取得章節的完整路徑，如 '4. 營運概況 > 4.2 富邦人壽營運概況'"""
        parts = []
        current_id = section.section_id

        # 逐層往上找父章節
        for sec in self.sections:
            if current_id.startswith(sec.section_id) and sec.section_id != current_id:
                # 檢查是否為直接祖先（如 "4" 是 "4.2" 的祖先）
                if self._is_ancestor(sec.section_id, current_id):
                    parts.append(f"{sec.section_id} {sec.title}")

        parts.append(f"{section.section_id} {section.title}")

        # 去重並保持順序
        seen = set()
        unique = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return " > ".join(unique)

    def _is_ancestor(self, ancestor_id: str, child_id: str) -> bool:
        """判斷 ancestor_id 是否為 child_id 的祖先"""
        ancestor_parts = ancestor_id.rstrip('.').split('.')
        child_parts = child_id.rstrip('.').split('.')
        if len(ancestor_parts) >= len(child_parts):
            return False
        return child_parts[:len(ancestor_parts)] == ancestor_parts

    def _find_toc_pages(self, doc) -> str:
        """自動偵測目錄頁並提取文字"""
        toc_texts = []

        for page_num in range(min(10, len(doc))):
            page = doc[page_num]
            text = page.get_text("text")

            # 目錄頁特徵：包含「目錄」字樣或有大量 dots + 數字格式的行
            lines = text.split("\n")
            dot_lines = sum(1 for l in lines if re.search(r'\.{5,}|…{3,}', l))

            if "目錄" in text[:50] or dot_lines > 5:
                toc_texts.append(text)

        return "\n".join(toc_texts)

    def _parse_toc_lines(self, toc_text: str) -> list[tuple]:
        """解析目錄文字行，返回 [(section_id, title, page_num, level), ...]"""
        results = []
        lines = toc_text.split("\n")

        # 預處理：合併分行的章節（如 "1.\n致股東報告書 .... 1"）
        merged_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 檢查是否為獨立的章節編號行（如 "1." 或 "3."）
            if re.match(r'^\d+\.\s*$', line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and re.search(r'\.{5,}|…{3,}', next_line):
                    merged_lines.append(f"{line} {next_line}")
                    i += 2
                    continue
            merged_lines.append(line)
            i += 1

        # 模式：章節編號 + 標題 + dots/空格 + 頁碼
        pattern = re.compile(
            r'^\s*'
            r'((?:\d+\.)+\s*\d*\.?)\s*'  # 章節編號：如 "2.1.1" 或 "4." 或 "1. "
            r'(.+?)'                       # 標題
            r'\s*[.…·]+\s*'                # 分隔 dots
            r'(\d+)\s*$'                   # 頁碼
        )

        for line in merged_lines:
            line = line.strip()
            if not line or line == "目錄":
                continue

            match = pattern.match(line)
            if match:
                sec_id = match.group(1).strip().rstrip('.')
                title = match.group(2).strip()
                page_num = int(match.group(3))
                level = len(sec_id.split('.'))

                # 過濾掉太長的「標題」（可能是解析錯誤）
                if len(title) < 80:
                    results.append((sec_id, title, page_num, level))

        return results

    def _detect_page_offset(self, doc, raw_sections: list) -> int:
        """自動偵測文件頁碼與實體頁碼的偏移量。
        offset 定義：physical_page(1-indexed) = doc_page + offset
        """
        if not raw_sections:
            return 0

        # 方法1：找到 "- 1 -" 頁碼標記，這是文件第1頁
        for phys_idx in range(len(doc)):
            page = doc[phys_idx]
            text = page.get_text("text")
            # 年報常見的頁碼格式：- 1 -（出現在頁面頂部或底部）
            if re.search(r'^\s*-\s*1\s*-\s*$', text, re.MULTILINE):
                # doc_page 1 → physical_page (phys_idx + 1)
                return (phys_idx + 1) - 1  # offset = phys_idx

        # 方法2：用目錄中的頂層章節標題在文件中搜尋
        for sec_id, title, doc_page, level in raw_sections[:5]:
            if level > 1:
                continue
            title_short = title[:8]
            for phys_idx in range(len(doc)):
                page = doc[phys_idx]
                text = page.get_text("text")
                # 搜尋章節標題（排除目錄頁本身）
                if title_short in text and "目錄" not in text[:50]:
                    # 確認是章節開頭（有編號前綴）
                    if re.search(rf'{re.escape(sec_id)}[\.\s]', text):
                        return (phys_idx + 1) - doc_page

        return 0

    def _compute_end_pages(self, total_pages: int):
        """計算每個章節的結束頁"""
        for i, section in enumerate(self.sections):
            # 找下一個同層級或更高層級的章節
            next_page = total_pages
            for j in range(i + 1, len(self.sections)):
                if self.sections[j].level <= section.level:
                    next_page = self.sections[j].physical_page - 1
                    break
            section.end_page = next_page

    def _build_page_mapping(self):
        """建立 physical_page → Section 映射（最細粒度的章節）"""
        self.page_to_section = {}

        # 由最細粒度的章節覆蓋到最粗粒度
        sorted_sections = sorted(self.sections, key=lambda s: (-s.level, s.physical_page))
        for section in sorted_sections:
            for page in range(section.physical_page, section.end_page + 1):
                if page not in self.page_to_section:
                    self.page_to_section[page] = section

        # 再用粗粒度章節填補空缺
        sorted_coarse = sorted(self.sections, key=lambda s: (s.level, s.physical_page))
        for section in sorted_coarse:
            for page in range(section.physical_page, section.end_page + 1):
                if page not in self.page_to_section:
                    self.page_to_section[page] = section
