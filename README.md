# Discord Gemini AI Bot 🤖

具備對話歷史紀錄持久化（SQLite）、多檔案動態系統提示詞（System Instructions）載入，以及安全的動態更新人設與清除對話歷史的 Discord Bot

---

## 功能

- **對話歷史持久化**：使用 SQLite 資料庫 `conversation.db` 紀錄每位使用者的對話歷史
- **模組化系統提示詞 (System Prompt)**：可從指定目錄（如 `system_instructions/`）載入多個檔案，並自動按照檔名排序拼接，方便您將基礎設定、人設、知識庫分開管理
- **動態更新人設**：透過 Discord 指令上傳檔案附件，即可即時更新機器人的人設
- **自動訊息分段**：當 Gemini 回應字數超過 Discord 的 2000 字限制時，Bot 會自動尋找適合的換行符號將訊息切片發送，避免訊息被強行截斷
- **日誌系統**：同時輸出至主控台與 `bot.log` 檔案，支援 `-d` / `--debug` 參數啟動詳細除錯日誌

---

## 系統要求

- **Python** >= 3.13
- **uv**

---

## 快速開始

### 1. 取得專案代碼

將此專案下載或克隆至本地目錄：

```bash
git clone https://github.com/qrspncpr/discord-llm.git
cd discord-llm
```

### 2. 配置環境變數

將專案目錄下的 `.env.simple` 複製並重命名為 `.env`：
```bash
cp .env.simple .env
```

打開 `.env` 並填入您的金鑰與偏好設定：
```ini
# Discord 機器人 Token (至 Discord Developer Portal 申請)
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# Gemini API 金鑰 (至 Google AI Studio 申請)
GEMINI_API_KEY=your_gemini_api_key_here

# 使用的 Gemini 模型名稱 (例如 gemini-2.0-flash-lite-001 或 gemini-2.0-flash)
MODEL=gemini-2.0-flash-lite-001

# 儲存系統提示詞的資料夾路徑
SYSTEM_INSTRUCTION_DIR=system_instructions

# SQLite 資料庫名稱
DB_NAME=conversation.db

# 每次對話時，載入並發送給 Gemini 的最大歷史紀錄筆數 (user + model 總和)
MAX_HISTORY_RECORDS=4

# 機器人指令前綴
COMMAND_PREFIX=!

# 聊天指令的別名，可以用 !gpt 或是 !llm 來觸發
ALIASES=["gpt", "llm"]

# Gemini 生成的溫度參數 (0.0 較嚴謹，1.0 較具創意)
TEMPERATURE=1
```

> [!WARNING]
> 請確保將 `.env` 加入到您的 `.gitignore` 中（本專案預設已忽略），切勿將敏感金鑰上傳至公開代碼庫

### 3. Discord 開發者設定

為了讓機器人能正常接收訊息並運作，請在 **[Discord Developer Portal](https://discord.com/developers/applications)** 進行以下設定：

1. 選擇您的 Application，點擊左側的 **Bot** 標籤頁
2. 向下滾動找到 **Privileged Gateway Intents** 區段
3. **必須啟用** **MESSAGE CONTENT INTENT**（將開關切換為啟用）
4. 儲存設定（Save Changes）
   > [!IMPORTANT]
   > 若未啟用 Message Content Intent，Bot 將無法讀取使用者的訊息內容，導致所有對話指令（如 `!bot`）失效

### 4. 設定系統提示詞 (System Instructions)

1. 在專案根目錄下建立一個名為 `system_instructions` 的資料夾（或您在 `.env` 中指定的目錄）
2. 在該資料夾內建立 `.txt` 檔案（例如 `01_base.txt`、`02_roleplay.txt`）
3. Bot 啟動時會自動讀取此資料夾內的所有檔案，並依檔名順序拼接成最終的 `SYSTEM_PROMPT`

### 5. 啟動機器人

#### 正常模式運行
```bash
uv run main.py
```

#### 除錯模式運行 (顯示更詳細的 DEBUG 日誌，方便排查問題)
```bash
uv run main.py --debug
# 或簡寫為
uv run main.py -d
```

---

## 機器人指令說明

| 指令 | 別名 | 說明 | 範例 |
| :--- | :--- | :--- | :--- |
| `!bot <訊息>` | `!gpt`, `!llm` (可在 .env 自訂) | 與 Gemini 進行對話，會自動帶入設定的人設與歷史紀錄 | `!bot 幫我寫一個 Python 泡沫排序法` |
| `!update_system_prompt` | `!up` | **動態更新系統人設**：需在發送此指令時，**附帶上傳新的人設文字檔 (.txt)**Bot 會自動下載並即時套用若更新失敗會自動還原 | *上傳新提示詞檔案並附加文字* `!up` |
| `!clear_my_history` | `!del` | **清除個人對話歷史**：僅將**發送指令之使用者**的對話歷史紀錄在資料庫中標記為已刪除（不影響其他使用者） | `!del` |
| `!clear_history` | `!ch` | **清除所有對話歷史**：將**資料庫中所有使用者**的對話歷史紀錄標記為已刪除 | `!ch` |

---

## 資料庫結構

本專案使用 SQLite 儲存對話紀錄，資料庫檔案預設為 `conversation.db`，其中的 `history` 資料表結構如下：

- `user_id` (INTEGER): Discord 使用者的唯一 ID
- `role` (TEXT): 角色類型，`user` 代表使用者發送的訊息，`model` 代表 Gemini 的回覆
- `message` (TEXT): 對話內容文字
- `token_count` (INTEGER): 該次對話消耗的總 Token 數（僅記錄在 `model` 角色行中）
- `timestamp` (DATETIME): 訊息產生的時間
- `is_deleted` (INTEGER): 軟刪除標記，`0` 表示正常，`1` 表示已被清除
