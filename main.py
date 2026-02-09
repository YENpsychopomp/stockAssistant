import os
import time
import json
import dotenv
import traceback
import uvicorn as uv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Any
import jwt
from decimal import Decimal
from fastapi import FastAPI, Request, Header
from fastapi.responses import StreamingResponse
from embedding import build_azure_client_from_env, ChromaDBManager, query
from db import connect_db, check_and_reset_user, deduct_quota
# LangChain
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()
TEST = False
OPENWEBUI_SECRET_KEY = dotenv.get_key(".env", "OPENWEBUI_SECRET_KEY")

class TokenCostTracker:
    def __init__(self, model_name: str, input_price_per_million: float, output_price_per_million: float, token_unit: int = 1000000):
        """
        初始化 Token 計算器\n
        他可以幫助我們在串流過程中累積計算 Token 用量和對應的成本。\n
        :param model_name: 模型名稱 (方便辨識)
        :param input_price_per_million: 每百萬 Input Token 的價格 (美金)
        :param output_price_per_million: 每百萬 Output Token 的價格 (美金)
        :param token_unit: Token 的單位，預設為 1,000,000 (百萬)
        """
        self.model_name = model_name
        self.input_price = Decimal(str(input_price_per_million))
        self.output_price = Decimal(str(output_price_per_million))
        self.token_unit = token_unit
        
        # 累積計數器
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = Decimal("0.0")

    def add_usage(self, input_tokens: int, output_tokens: int):
        """累加當次呼叫的 Token 用量"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # 計算當次花費
        input_cost = (Decimal(input_tokens) / Decimal(self.token_unit)) * self.input_price
        output_cost = (Decimal(output_tokens) / Decimal(self.token_unit)) * self.output_price
        self.total_cost += (input_cost + output_cost)

    def get_summary(self) -> str:
        """回傳格式化的統計字串"""
        return (
            f"\n[Cost] Model: {self.model_name}\n"
            f"   - Input: {self.total_input_tokens} | Output: {self.total_output_tokens}\n"
            f"   - Total Cost: ${self.total_cost:.6f}"
        )

# ==========================================
# 第一步：定義工具 (Tools)
# 使用 @tool 裝飾器，LangChain 會自動把 docstring 轉成 OpenAI 看得懂的 schema
# ==========================================

@tool
def getPriceTool(symbol: str) -> str:
    """
    取得股票即時價格。當用戶詢問「目前股價多少？」、「現在價格？」時，請呼叫此工具。
    參數 symbol: 股票代號，例如 '2330.TW' 或 'AAPL'。
    當用戶只提供數字代碼時，請自動補上 .TW (台股)。
    例如，輸入 '2330'，則轉換為 '2330.TW'。
    這樣可以確保台股代碼的正確性。
    """
    print("call get price tool")
    # 1. 自動修正台股代碼
    # 如果是純數字且長度為 4-6 位，自動補上 .TW (這能覆蓋大多數台股)
    clean_symbol = symbol.strip().upper()
    if clean_symbol.isdigit():
        clean_symbol = f"{clean_symbol}.TW"
        print(f"自動修正代碼: {symbol} -> {clean_symbol}")

    try:
        ticker = yf.Ticker(clean_symbol)
        data = ticker.info
        
        # 取得現價 (處理多種可能的 Key)
        current_price = data.get('currentPrice') or data.get('regularMarketPrice')
        
        if not current_price:
            return f"找不到 {clean_symbol} 的價格數據，請確認代碼是否正確（台股請加 .TW）。"

        # 獲取月表現
        hist = ticker.history(period="1mo")
        if not hist.empty:
            perf_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            performance_str = f"{perf_pct:.2f}%"
        else:
            performance_str = "無法取得"

        return (f"{data.get('shortName', clean_symbol)} 的當前股價為 ${current_price}，"
                f"過去一個月的表現為 {performance_str}。")
            
    except Exception as e:
        return f"讀取 {clean_symbol} 時發生錯誤: {str(e)}"

@tool
def rag_tool(text: str) -> str:
    """
    檢索內部的財經研究報告或新聞。
    當使用者問「某公司發生什麼事」或「最新新聞」時使用。
    """
    print(f"[Tool] 正在檢索 RAG 資料庫: {text}...")
    azure_client = build_azure_client_from_env()
    stockNewsChromaDB = ChromaDBManager(collection_name="stock_news", persist_dir="db/chroma_db")
    result = query(Azure_client=azure_client, ChromaDB=stockNewsChromaDB, query=text, top_k=5)
    return result

@tool
def technicalAnalysis(symbol: str) -> str:
    """
    取得股票的技術分析指標，包括 RSI、MACD、布林通道等，並給出簡單的趨勢判斷。
    當使用者問「趨勢如何？」、「強勢嗎？」、「技術分析」時使用。
    """
    symbol = symbol.strip().upper()

    if symbol.isdigit():
        target_symbol = f"{symbol}.TW"
    else:
        target_symbol = symbol

    try:
        ticker = yf.Ticker(target_symbol)
        df = ticker.history(period="6mo")

        if df.empty:
            search = yf.Search(symbol, max_results=1)
            if not search.quotes:
                return f"找不到與 '{symbol}' 相關的股票資料"
            target_symbol = search.quotes[0]["symbol"]
            ticker = yf.Ticker(target_symbol)
            df = ticker.history(period="6mo")

        bbands = df.ta.bbands(length=20)
        rsi = df.ta.rsi(length=14)
        macd = df.ta.macd()

        if bbands is None or rsi is None or macd is None:
            return f"{target_symbol} 資料不足，無法計算指標"

        df = pd.concat([df, bbands, rsi, macd], axis=1)

        latest = df.iloc[-1]

        rsi_col = [c for c in df.columns if c.startswith("RSI_")][0]
        macd_col = [c for c in df.columns if c.startswith("MACD_") and not c.endswith("h")][0]
        signal_col = [c for c in df.columns if c.startswith("MACDs_")][0]
        bbu_col = [c for c in df.columns if c.startswith("BBU_")][0]
        bbl_col = [c for c in df.columns if c.startswith("BBL_")][0]

        rsi_val = latest[rsi_col]
        macd_val = latest[macd_col]
        signal_val = latest[signal_col]

        if rsi_val < 30 and macd_val > signal_val:
            suggestion = "偏多訊號（可能反彈）"
        elif rsi_val > 70 and macd_val < signal_val:
            suggestion = "偏空訊號（注意回檔）"
        else:
            suggestion = "區間整理"

        return (
            f"【{target_symbol} 技術分析】\n"
            f"- 收盤價: {latest['Close']:.2f}\n"
            f"- RSI(14): {rsi_val:.2f}\n"
            f"- 布林下軌: {latest[bbl_col]:.2f}\n"
            f"- 布林上軌: {latest[bbu_col]:.2f}\n"
            f"- 判斷: {suggestion}"
        )

    except Exception as e:
        return f"分析失敗: {e}"

SYSTEM_PROMPT = (
    "# Role 你是一個專業財經助手 StockAssistant。\n"
    "1. 查詢即時股價 (getPriceTool)\n"
    "2. 檢索內部的財經研究報告或新聞新聞 (RAGTool)\n"
    "3. 執行 K 線與技術指標分析 (technicalAnalysis)\n\n"
    "# Rules\n"
    "- **台股代碼**：請務必提醒自己或使用者，台股必須加 `.TW` (上市) 或 `.TWO` (上櫃)。\n"
    "- **工具選擇**：\n"
    "- 詢問「某公司最近發生什麼事？」 -> 優先使用 `RAGTool`。\n"
    "- 詢問「現在價錢多少？」 -> 使用 `getPriceTool`。\n"
    "- 詢問「趨勢如何、強勢嗎？」 -> 使用 `technicalAnalysis`。\n"
)

# ==========================================
# 第二步：建立大腦 (Model + Agent)
# ==========================================

llm = AzureChatOpenAI(
    azure_deployment=dotenv.get_key(".env", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    azure_endpoint=dotenv.get_key(".env", "AZURE_OPENAI_ENDPOINT"),
    api_version=dotenv.get_key(".env", "OPENAI_API_VERSION"),
    api_key=dotenv.get_key(".env", "AZURE_OPENAI_API_KEY"),
    streaming=True,
)

agent = create_agent(
    model=llm,
    tools=[getPriceTool, rag_tool, technicalAnalysis],
    system_prompt=SYSTEM_PROMPT,
)

def _usage_payload(cost_tracker: TokenCostTracker) -> dict:
    """取得目前的 Token 統計"""
    return {
        "prompt_tokens": cost_tracker.total_input_tokens,
        "completion_tokens": cost_tracker.total_output_tokens,
        "total_tokens": cost_tracker.total_input_tokens + cost_tracker.total_output_tokens,
        "cost_usd": float(cost_tracker.total_cost), 
    }

async def stream_agent_query(user_input: str, history: list, cost_tracker: TokenCostTracker):
    state = {"messages": [*history, HumanMessage(content=user_input)]}
    
    try:
        # 使用 async for 直接迭代 agent 的串流
        async for mode, chunk in agent.astream(state, stream_mode=["messages", "updates"]):
            
            # -------------------------------------------------
            # 修正點：處理 'messages' 模式下的 Tuple
            # -------------------------------------------------
            if mode == "messages":
                # 預設 message 為 chunk 本身
                msg = chunk
                
                # 如果是 Tuple (根據你的 Log，這裡是關鍵)，取出第一個元素
                if isinstance(chunk, tuple):
                    msg = chunk[0]
                
                # 現在 msg 應該是 AIMessageChunk 了，檢查有沒有 content
                if hasattr(msg, "content"):
                    # 有些特殊狀況 content 是 list (多模態)，轉成字串防呆
                    text = msg.content
                    if isinstance(text, list):
                        text = "".join([str(x) for x in text])
                    
                    if text:
                        yield text

            # -------------------------------------------------
            # 處理 'updates' (修正後的 Usage 提取路徑)
            # -------------------------------------------------
            elif mode == "updates":
                if isinstance(chunk, dict):
                    usage = None
                    
                    # 根據你提供的 Log 結構：chunk -> 'model' -> 'messages' -> AIMessage
                    if "model" in chunk and "messages" in chunk["model"]:
                        messages = chunk["model"]["messages"]
                        if messages:
                            # 取得最後一條訊息（通常是 AIMessageChunk 或 AIMessage）
                            last_msg = messages[-1]
                            # 使用 getattr 安全取得 usage_metadata
                            usage = getattr(last_msg, "usage_metadata", None)
                    
                    # 備份路徑 (有些版本直接在 chunk 裡)
                    elif "usage_metadata" in chunk:
                        usage = chunk["usage_metadata"]

                    if usage:
                        print(f"[成功抓取] Usage: {usage}")
                        cost_tracker.add_usage(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )

    except Exception as e:
        print(f"發生錯誤: {e}")
        traceback.print_exc()

# ==========================================
# 3. FastAPI Endpoint (OpenAI 格式)
# ==========================================
@app.post("/v1/chat/completions")
async def chat_completions(request_body: dict, request: Request):
    # --- A. 身分驗證與 ID 解析 ---
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    
    user_id = "unknown_user"
    try:
        # 嘗試解碼 JWT 取得真實 User ID
        payload = jwt.decode(token, OPENWEBUI_SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("id")
    except Exception as e:
        print(f"⚠️ JWT 解析失敗 (可能是測試模式或 Key 錯誤): {e}")
        # 如果是測試模式，可以使用預設 ID，否則應回傳 401
        if TEST:
            user_id = "test_user_001"
        else:
            return {"error": "Invalid Authentication"}

    # --- B. 流量控制：檢查與重置 ---
    # 開啟連線進行檢查
    conn = connect_db() 
    try:
        # 這一行會自動處理：1.新用戶註冊 2.超過24小時自動重置 3.回傳目前餘額
        current_quota = check_and_reset_user(conn, user_id)
    finally:
        conn.close() # 檢查完立刻關閉連線

    # 如果餘額不足，直接拒絕請求
    if current_quota <= 0:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "您的額度已用完，請等待 24 小時後重置。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # --- C. 準備 Agent 輸入 ---
    message_list = request_body.get("messages", [])
    user_input = message_list[-1].get("content", "") if message_list else ""
    
    history = []
    msg_index = len(message_list) - 2
    while len(history) < 4 and msg_index >= 0:
        content = message_list[msg_index].get("content", "")
        if message_list[msg_index].get("role") == "user":
            history.append(HumanMessage(content=content))
        elif message_list[msg_index].get("role") == "assistant":
            if content == "您的額度已用完，請等待 24 小時後重置。":
                history.pop()  # 把上一輪的 AI 回覆（提示用戶額度不足）從歷史中移除，避免干擾 Agent 判斷
            else:
                history.append(AIMessage(content=content))
        msg_index -= 1
        history.reverse()
    # --- D. 定義串流生成器 (包含扣款邏輯) ---
    async def response_generator():
        # 初始化計費器
        cost_tracker = TokenCostTracker(
            model_name="gpt-4.1-nano", 
            input_price_per_million=0.1,
            output_price_per_million=0.4
        )
        
        chat_id = f"chatcmpl-{int(time.time())}"
        created_time = int(time.time())

        # 1. 開始串流 Agent 回覆
        async for text_chunk in stream_agent_query(user_input, history, cost_tracker=cost_tracker):
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "gpt-4.1-nano",
                "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        
        # 2. 串流結束，準備結算
        usage_data = _usage_payload(cost_tracker)
        total_cost = usage_data.get("cost_usd", 0.0)
        
        if total_cost > 0:
            deduct_conn = connect_db()
            try:
                deduct_quota(deduct_conn, user_id, float(total_cost))
            except Exception as e:
                print(f"❌ 扣款失敗: {e}")
            finally:
                deduct_conn.close()

        # 3. 發送 Usage 資訊給前端
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "gpt-4.1-nano",
            "choices": [],
            "usage": usage_data
        }
        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # --- F. 回傳回應 ---
    return StreamingResponse(
        response_generator(), 
        media_type="text/event-stream"
    )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "financial-rag-assistant", # 讓 Open WebUI 選單看得到的 ID
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom-proxy"
            }
        ]
    }

if __name__ == "__main__":
    uv.run("main:app", host="127.0.0.1", port=9898, reload=True)