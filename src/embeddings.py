import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embeddings():
    """文件端 Embedding（用於建立索引）"""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )


def get_query_embeddings():
    """查詢端 Embedding（用於搜尋）"""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
        task_type="question_answering"
    )


def embed_page_image(image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
    """用 gemini-embedding-2-preview 嵌入頁面圖片（跨模態）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.embed_content(
        model='gemini-embedding-2-preview',
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
    )
    return result.embeddings[0].values


def embed_page_images_batch(image_list: list[bytes], mime_type: str = "image/png") -> list[list[float]]:
    """批次嵌入多張頁面圖片（每次最多 6 張）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    all_embeddings = []

    # gemini-embedding-2-preview 每次最多 6 張圖片
    batch_size = 6
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        contents = [
            types.Part.from_bytes(data=img, mime_type=mime_type)
            for img in batch
        ]
        result = client.models.embed_content(
            model='gemini-embedding-2-preview',
            contents=contents
        )
        for emb in result.embeddings:
            all_embeddings.append(emb.values)

    return all_embeddings
