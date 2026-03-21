import os
import hashlib
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embeddings import get_embeddings
from config import VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


class VectorStoreManager:
    """管理 FAISS 向量資料庫，支援文字索引 + 圖片索引雙通道"""

    def __init__(self):
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    def build_index(self, documents: list[Document], pdf_path: str) -> FAISS:
        """為 PDF 建立文字 FAISS 索引"""
        store_id = self._get_store_id(pdf_path)
        store_path = os.path.join(VECTORSTORE_DIR, store_id)

        # 若已存在索引則直接載入
        if os.path.exists(store_path):
            return self._load_from_path(store_path)

        # 建立新索引
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(store_path)
        return vectorstore

    def build_image_index(self, pdf_path: str) -> dict:
        """為 PDF 建立頁面圖片 FAISS 索引（返回 {faiss_index, page_numbers}）"""
        from config import PAGE_IMAGES_DIR

        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        img_dir = os.path.join(PAGE_IMAGES_DIR, basename)
        cache_path = os.path.join(VECTORSTORE_DIR, f"img_{self._get_store_id(pdf_path)}")

        # 檢查快取
        if os.path.exists(f"{cache_path}.npy"):
            vectors = np.load(f"{cache_path}.npy")
            page_numbers = np.load(f"{cache_path}_pages.npy")
            return {"vectors": vectors, "page_numbers": page_numbers.tolist()}

        if not os.path.exists(img_dir):
            return {"vectors": None, "page_numbers": []}

        # 讀取所有頁面圖片
        from src.embeddings import embed_page_images_batch

        image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        if not image_files:
            return {"vectors": None, "page_numbers": []}

        print(f"[ImageIndex] 正在嵌入 {len(image_files)} 頁圖片...")
        image_bytes_list = []
        page_numbers = []
        for f in image_files:
            with open(os.path.join(img_dir, f), 'rb') as fh:
                image_bytes_list.append(fh.read())
            page_numbers.append(int(f.split('_')[1].split('.')[0]))

        try:
            embeddings = embed_page_images_batch(image_bytes_list)
            vectors = np.array(embeddings, dtype=np.float32)

            # 儲存快取
            np.save(f"{cache_path}.npy", vectors)
            np.save(f"{cache_path}_pages.npy", np.array(page_numbers))

            print(f"[ImageIndex] 完成！{vectors.shape[0]} 頁, {vectors.shape[1]} 維度")
            return {"vectors": vectors, "page_numbers": page_numbers}
        except Exception as e:
            print(f"[ImageIndex] 圖片索引建立失敗: {e}")
            return {"vectors": None, "page_numbers": []}

    def load_index(self, pdf_path: str) -> FAISS:
        """載入已存在的文字索引"""
        store_id = self._get_store_id(pdf_path)
        store_path = os.path.join(VECTORSTORE_DIR, store_id)
        return self._load_from_path(store_path)

    def merge_indexes(self, pdf_paths: list[str]) -> FAISS:
        """合併多個 PDF 的 FAISS 索引為單一 vectorstore"""
        if not pdf_paths:
            raise ValueError("至少需要一個 PDF")
        base_vs = self.load_index(pdf_paths[0])
        for path in pdf_paths[1:]:
            other_vs = self.load_index(path)
            base_vs.merge_from(other_vs)
        return base_vs

    def _load_from_path(self, store_path: str) -> FAISS:
        """從路徑載入索引"""
        embeddings = get_embeddings()
        return FAISS.load_local(store_path, embeddings,
                                allow_dangerous_deserialization=True)

    def index_exists(self, pdf_path: str) -> bool:
        """檢查文字索引是否存在"""
        store_id = self._get_store_id(pdf_path)
        store_path = os.path.join(VECTORSTORE_DIR, store_id)
        return os.path.exists(store_path)

    def _get_store_id(self, pdf_path: str) -> str:
        """store ID = file_hash + config_hash，參數變更自動重建"""
        file_hash = self._get_file_hash(pdf_path)
        config_hash = self._get_config_hash()
        return f"{file_hash}_{config_hash}"

    def _get_file_hash(self, filepath: str) -> str:
        """計算檔案 hash"""
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:12]

    def _get_config_hash(self) -> str:
        """計算 config hash（chunk_size + chunk_overlap + embedding_model）"""
        config_str = f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBEDDING_MODEL}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_file_hash(self, filepath: str) -> str:
        """向下相容"""
        return self._get_file_hash(filepath)
