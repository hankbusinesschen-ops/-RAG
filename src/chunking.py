from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class SmartChunker:
    """智慧分塊：結構感知（TOC 章節邊界）+ 中文友好分隔符 + 表格保留完整"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.toc_parser = None  # 延遲注入

    def set_toc_parser(self, toc_parser):
        """注入 TOC 解析器（在 parse 完 PDF 後設定）"""
        self.toc_parser = toc_parser

    def chunk_documents(self, documents: list[Document]) -> tuple[list[Document], dict]:
        """
        對文件列表進行智慧分塊。
        Returns:
            chunks: 分塊後的文件列表
            parent_docs: {page_num: 完整頁面文字} 用於 Parent Document Retriever
        """
        all_chunks = []
        parent_docs = {}

        for doc in documents:
            page = doc.metadata["page"]
            source = doc.metadata.get("source", "unknown")
            # 建立 parent 文件（完整頁面內容），用 (source, page) 避免多文件頁碼衝突
            parent_key = (source, page)
            if parent_key not in parent_docs:
                parent_docs[parent_key] = doc.page_content
            else:
                parent_docs[parent_key] += "\n\n" + doc.page_content

            if doc.metadata.get("type") == "table":
                # 表格不切割，完整保留，加上章節前綴
                enriched_doc = self._enrich_with_section(doc)
                all_chunks.append(enriched_doc)
            else:
                chunks = self._split_text_doc(doc)
                all_chunks.extend(chunks)

        # 為每個 chunk 指派 ID
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"
            chunk.metadata["parent_page"] = chunk.metadata["page"]

        return all_chunks, parent_docs

    def _split_text_doc(self, doc: Document) -> list[Document]:
        """文字分塊，使用中文友好的分隔符，前綴章節路徑"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # 段落
                "\n",     # 換行
                "。",     # 中文句號
                "；",     # 中文分號
                "，",     # 中文逗號
                " ",
            ],
            length_function=len,
        )
        texts = splitter.split_text(doc.page_content)

        chunks = []
        for i, t in enumerate(texts):
            chunk_doc = Document(
                page_content=t,
                metadata={**doc.metadata, "chunk_index": i}
            )
            # 加上章節前綴和 metadata
            chunk_doc = self._enrich_with_section(chunk_doc)
            chunks.append(chunk_doc)

        return chunks

    def _enrich_with_section(self, doc: Document) -> Document:
        """為 chunk 添加章節資訊（前綴 + metadata）"""
        if not self.toc_parser:
            return doc

        page = doc.metadata["page"]
        section_info = self.toc_parser.get_section_for_page(page)

        if not section_info:
            return doc

        # 添加 metadata
        doc.metadata["section_id"] = section_info["section_id"]
        doc.metadata["section_title"] = section_info["section_title"]
        doc.metadata["section_path"] = section_info["section_path"]

        # 在 chunk 內容前加上章節路徑前綴（幫助 embedding 捕捉語意）
        section_prefix = f"[{section_info['section_path']}]\n"
        if not doc.page_content.startswith("["):
            doc.page_content = section_prefix + doc.page_content

        return doc
