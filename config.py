import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model settings
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
LLM_MODEL = "gemini-3-flash-preview"
VISION_MODEL = "gemini-3-flash-preview"  # Vision 提取用模型

# Vision fallback settings
ENABLE_VISION_FALLBACK = True
VISION_CACHE_DIR = "data/vision_cache"
PAGE_IMAGES_DIR = "data/page_images"
MIN_CHINESE_CHARS = 50  # 提取中文字元數低於此閾值觸發 Vision fallback

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
RETRIEVE_K = 20        # 初始檢索數量（FAISS + BM25 各取 RETRIEVE_K）
RERANK_TOP_K = 5       # Reranking 後送入 LLM 的數量
TOP_K = 12             # 向下相容（未啟用 reranker 時的 fallback）
BM25_WEIGHT = 0.4
FAISS_WEIGHT = 0.6
RRF_K = 60

# Reranking settings
ENABLE_RERANKER = True  # 使用 Gemini LLM reranking（零本地記憶體）

# Hallucination detection
RELEVANCE_THRESHOLD = 1.2  # FAISS L2 distance threshold (lower = more similar)
ENABLE_GROUNDING_CHECK = True  # 啟用 LLM-based grounding 驗證

# Paths
UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"
VECTORSTORE_DIR = "vectorstore"
QA_PAIRS_PATH = "evaluation/qa_pairs.json"
