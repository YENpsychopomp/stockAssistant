"""Utilities for preparing text and writing embeddings to ChromaDB."""
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import chromadb
from bs4 import BeautifulSoup
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import tiktoken
import time
from pydantic import BaseModel

dotenv.load_dotenv()

PERSIST_DIR = "db/chroma_db"
COLLECTION_NAME = "articles"
DEFAULT_MODEL = "text-embedding-ada-002"
AZURE_BATCH_LIMIT = 16

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    stream: bool = False
# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------
class ChromaDBManager:
    """Small convenience wrapper for Chroma client/collection lifecycle."""

    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = self._init_client()
        self.collection = self._init_collection()

    def _init_client(self) -> ClientAPI:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _init_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store vectors and optional metadata/documents in the managed collection."""
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query_by_embedding(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the collection using a vector embedding."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None
        )

    def delete_all(self) -> None:
        """Remove every item in the managed collection."""
        self.collection.delete(where={})

    def count(self) -> int:
        """Return number of items currently stored."""
        return self.collection.count()

    def existing_ids(self, ids: List[str]) -> Set[str]:
        """Return ids that already exist in the collection."""
        if not ids:
            return set()
        result = self.collection.get(ids=ids, include=[])
        return set(result.get("ids", []))


# ---------------------------------------------------------------------------
# Budget utilities
# ---------------------------------------------------------------------------
class BudgetManager:
    """Simple budget guard for embedding batches."""

    def __init__(self, model_name: str = DEFAULT_MODEL, price_per_1k_tokens: float = 0.0001, budget_usd: float = 1.0):
        self.model_name = model_name
        self.price_rate = price_per_1k_tokens
        self.budget = budget_usd
        self.encoder = tiktoken.encoding_for_model(model_name)

    def calculate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def check_budget(self, texts: Sequence[str]) -> tuple[int, float]:
        total_tokens = sum(self.calculate_tokens(t) for t in texts)
        estimated_cost = (total_tokens / 1000) * self.price_rate

        print("--- 預算檢查 ---")
        print(f"預計消耗 Tokens: {total_tokens}")
        print(f"預計花費: ${estimated_cost:.6f} USD")
        print(f"剩餘預算: ${self.budget:.6f} USD")

        if estimated_cost > self.budget:
            raise ValueError(f"預算不足！預計花費 ${estimated_cost:.6f} > 預算 ${self.budget:.6f}")

        return total_tokens, estimated_cost


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_html_to_markdown(raw_html: str) -> str:
    """Convert HTML into a Markdown-ish, embedding-friendly string."""
    decoded_html = html.unescape(raw_html)
    soup = BeautifulSoup(decoded_html, "html.parser")

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append("| " + " | ".join(cols) + " |")
        markdown_table = "\n" + "\n".join(rows) + "\n"
        table.replace_with(markdown_table)

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    cleaned_text = re.split(r"原始連結|看更多|延伸閱讀", cleaned_text)[0]
    return cleaned_text

def _batch(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------
def run_embedding_pipeline(
    documents: Sequence[Dict[str, Any]],
    azure_client: AzureOpenAI,
    budget_manager: BudgetManager,
    model_name: str = DEFAULT_MODEL,
) -> Optional[List[Dict[str, Any]]]:
    processed_texts: List[str] = [doc["text"] for doc in documents]
    metadata_list: List[Dict[str, Any]] = [doc.get("metadata", {}) for doc in documents]
    ids: List[str] = [str(doc.get("id", idx)) for idx, doc in enumerate(documents)]

    print(">>> 2. 進行預算評估...")
    try:
        budget_manager.check_budget(processed_texts)
    except ValueError as exc:  # keep going gracefully
        print(exc)
        return None

    print(">>> 3. 呼叫 Azure API 進行 Embedding...")
    try:
        embeddings: List[List[float]] = []
        for batch in _batch(processed_texts, AZURE_BATCH_LIMIT):
            response = azure_client.embeddings.create(input=batch, model=model_name)
            embeddings.extend([data.embedding for data in response.data])

        print(f"成功生成 {len(embeddings)} 筆向量資料")

        return [
            {"id": ids[idx], "text": text, "metadata": metadata_list[idx], "vector": embeddings[idx]}
            for idx, text in enumerate(processed_texts)
        ]
    except Exception as exc:  # surface API errors for debugging
        print(f"API 呼叫失敗: {exc}")
        return None

def process_auto_save(ai_reply: str):
    """提取 JSON 並存入 ChromaDB"""
    try:
        raw_json = ai_reply.split("[SAVE_START]")[1].split("[SAVE_END]")[0].strip()
        data = json.loads(raw_json)
        print(f"[自動入庫] 標題: {data.get('title')}")
        
        # 這裡可以根據你之前 embedding.py 的邏輯存入
        # 為了簡化，你可以直接用 db_manager.collection.add (...)
        # 或調用你原本寫好的批次處理函數
    except Exception as e:
        print(f"入庫失敗: {e}")

def build_azure_client_from_env() -> AzureOpenAI:
    api_key = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_KEY")
    endpoint = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_ENDPOINT")
    api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_VERSION")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

def query(Azure_client: AzureOpenAI, ChromaDB: ChromaDBManager, query: str, top_k: int = 5) -> str:
    """簡單查詢介面"""
    mb_res = Azure_client.embeddings.create(input=[query], model="text-embedding-ada-002")
    q_emb = mb_res.data[0].embedding
    search_res = ChromaDB.query_by_embedding(query_embedding=q_emb, top_k=top_k)
    context = ""
    if search_res['documents'] and search_res['documents'][0]:
        context = "\n".join(search_res['documents'][0])
    return context

if __name__ == "__main__":
    stockNewsChromaDB = ChromaDBManager(collection_name="stock_news", persist_dir=PERSIST_DIR)
    client = build_azure_client_from_env()
    user_query = input("請輸入查詢內容: ").strip()
    print(f"正在查詢與 '{user_query}' 相關的文章...")
    a = client.embeddings.create(input=[user_query], model="text-embedding-ada-002")
    query_result = query(client, stockNewsChromaDB, user_query, top_k=5)
    print("查詢結果:")
    print(query_result)