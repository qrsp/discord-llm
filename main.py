import datetime
import json
import os
import sqlite3
import traceback

import discord
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# 載入 .env 檔案中的環境變數
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL")
SYSTEM_INSTRUCTION_FILE = os.getenv("SYSTEM_INSTRUCTION_FILE")
_ = os.getenv("ALIASES", "[]")
ALIASES = json.loads(_)
DB_NAME = os.getenv("DB_NAME", "conversation.db")
_ = os.getenv("MAX_HISTORY_RECORDS", "4") # 每個使用者最多儲存X則紀錄 (user + model)
MAX_HISTORY_RECORDS = int(_)
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")
_ = os.getenv("TEMPERATURE", "1")
TEMPERATURE = int(_)

# 檢查金鑰是否存在
if not DISCORD_BOT_TOKEN or not GEMINI_API_KEY or not MODEL:
    print("錯誤：請在 .env 檔案中設定 DISCORD_BOT_TOKEN、 GEMINI_API_KEY 和 MODEL")
    exit()

SYSTEM_PROMPT = ""

# 設定 Discord Bot 的權限 (Intents)
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

# 初始化 Bot
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# 設定 Gemini API
genai_client = genai.Client(api_key=GEMINI_API_KEY)

config = types.GenerateContentConfig(
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY ,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH ,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT ,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT ,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT ,
            threshold=types.HarmBlockThreshold.OFF,
        ),
    ],
    # max_output_tokens=1024,
    temperature=TEMPERATURE,
)

def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.replace(tzinfo=None).isoformat()

def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("datetime", convert_datetime)

def setup_database():
    """初始化資料庫和資料表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            token_count INTEGER,
            timestamp DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_history(user_id: int, role: str, message: str, token_count: int = 0):
    """新增一筆對話紀錄到資料庫"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (user_id, role, message, token_count, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, role, message, token_count, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_user_history(user_id: int, limit: int = 10) -> list:
    """從資料庫獲取指定使用者的歷史訊息"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT role, message FROM history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, limit))
    # 將歷史紀錄由舊到新排序，以符合對話順序
    history = reversed(cursor.fetchall())
    conn.close()

    # 轉換成 Gemini 需要的格式
    gemini_history = []
    for role, message in history:
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message)]
            ),
        )
    return gemini_history

# --- 人設 (System Prompt) 輔助函式 ---
def load_system_prompt():
    """從 character.txt 載入系統提示"""
    global SYSTEM_PROMPT
    try:
        with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
            SYSTEM_PROMPT = f.read().strip()
        print(f"成功載入SYSTEM_PROMPT: {SYSTEM_PROMPT[:50]}...")
    except FileNotFoundError:
        print(f"警告: {SYSTEM_INSTRUCTION_FILE} 不存在。")


class FileWatcher():
    def __enter__(self) -> None:
        event_handler = self.FileChangeHandler()
        self.observer = Observer()
        self.observer.schedule(event_handler, path='.', recursive=False)
        self.observer.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.observer.stop()
        self.observer.join()

    # 檔案監控事件處理器
    class FileChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if os.path.abspath(event.src_path) == os.path.abspath(SYSTEM_INSTRUCTION_FILE):
                load_system_prompt()


# --- Bot 事件監聽 ---
@bot.event
async def on_ready():
    """Bot 啟動時執行的動作"""
    setup_database()
    load_system_prompt()
    print(f'Bot 已登入為 {bot.user}')
    print('------')

# --- Bot 指令 ---
@bot.command(name='bot', aliases=ALIASES)
async def chat_command(ctx: commands.Context, *, message: str):
    """與 Gemini AI 進行對話"""
    user_id = ctx.author.id

    async with ctx.typing():
        try:
            # 取得歷史紀錄
            history = get_user_history(user_id)
            history.append(
                types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=message)]
                ),
            )
            config.system_instruction = SYSTEM_PROMPT

            response = genai_client.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
            print(response)

            # 儲存對話紀錄
            # 使用者訊息
            add_history(user_id, 'user', message)
            # 模型回應
            # response.usage_metadata.total_token_count 包含了提示和回應的 token 總數
            token_count = response.usage_metadata.total_token_count
            add_history(user_id, 'model', response.text, token_count)

            # 回傳結果到 Discord
            chunks = split_string_by_length_and_newline(response.text, max_len=2000)
            for chunk in chunks:
                await ctx.reply(chunk)

        except ServerError as e:
            print(f"ServerError：{e}")
            await ctx.reply(f"ServerError：{e}")
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(f"糟糕，發生錯誤了：{e}")

def split_string_by_length_and_newline(s, max_len):
    result = []
    start = 0
    while start < len(s):
        # 嘗試切到 max_len 範圍內最後一個 \n 的位置
        end = start + max_len
        if end >= len(s):
            chunk = s[start:]
            result.append(chunk)
            break

        sub = s[start:end]
        last_newline = sub.rfind('\n')

        if last_newline == -1:
            # 改為強制切斷，不保證以 \n 結尾
            last_newline = len(sub) - 1  # 或者找其他替代策略

        chunk = s[start:start + last_newline + 1]
        if chunk.strip():
            result.append(chunk)
        result.append(chunk)
        start += last_newline + 1

    return result

if __name__ == "__main__":
    if SYSTEM_INSTRUCTION_FILE:
        with FileWatcher():
            bot.run(DISCORD_BOT_TOKEN)
    else:
        bot.run(DISCORD_BOT_TOKEN)
