import sqlite3
import time

PERSON_USD_QUOTA_PER_DAY = 1.0
DB_PATH = 'users.db'

def init_db():
    """初始化資料庫 (如果不存在則建立)"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # reset_time: 記錄上一次「重置額度」的時間點 (Unix Timestamp)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                quota REAL,
                reset_time REAL
            )
        ''')
        conn.commit()

def connect_db():
    return sqlite3.connect(DB_PATH)

def check_and_reset_user(conn, user_id) -> float:
    """
    核心邏輯：
    1. 如果是新使用者 -> 建立並給額度，記錄現在時間。
    2. 如果是舊使用者 -> 檢查是否超過 24 小時。
       - 是 -> 重置額度，更新時間為現在。
       - 否 -> 什麼都不做，回傳目前餘額。
    回傳: 目前可用餘額 (float)
    """
    cursor = conn.cursor()
    current_time = time.time()
    one_day_seconds = 86400  # 24 小時

    # 1. 查詢使用者目前的額度與重置時間
    cursor.execute("SELECT quota, reset_time FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()

    # --- 情況 A: 新使用者 (Insert) ---
    if row is None:
        print(f"🆕 新使用者 {user_id}: 初始化額度與時間")
        cursor.execute(
            "INSERT INTO users (user_id, quota, reset_time) VALUES (?, ?, ?)", 
            (user_id, PERSON_USD_QUOTA_PER_DAY, current_time)
        )
        conn.commit()
        return PERSON_USD_QUOTA_PER_DAY

    quota, last_reset_time = row
    
    # 檢查是否超過 24 小時 (滑動窗口邏輯)
    if current_time - last_reset_time >= one_day_seconds:
        print(f"🔄 使用者 {user_id}: 已過 24 小時，重置額度")
        cursor.execute(
            "UPDATE users SET quota=?, reset_time=? WHERE user_id=?", 
            (PERSON_USD_QUOTA_PER_DAY, current_time, user_id)
        )
        conn.commit()
        return PERSON_USD_QUOTA_PER_DAY
    
    # --- 情況 C: 未過期，回傳剩餘額度 ---
    return quota

def deduct_quota(conn, user_id, cost: float):
    """
    直接在資料庫中扣除額度 (原子操作)
    """
    cursor = conn.cursor()
    # 使用 MAX(0, ...) 確保不會扣到變成負數
    cursor.execute(
        "UPDATE users SET quota = MAX(0, quota - ?) WHERE user_id=?", 
        (cost, user_id)
    )
    conn.commit()
    print(f"使用者 {user_id} 扣除 ${cost:.6f}")

init_db()