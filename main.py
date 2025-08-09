import argparse
import datetime
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import traceback
import aiohttp

import discord
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import aliases
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# 解析命令行參數
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Discord Bot with Gemini AI integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  uv run main.py              # 正常模式運行
  uv run main.py -d           # 除錯模式運行
  uv run main.py --debug      # 除錯模式運行
        '''
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='啟用除錯模式，顯示詳細的日誌訊息'
    )
    return parser.parse_args()

# 載入 .env 檔案中的環境變數
load_dotenv()

# 解析命令行參數
args = parse_arguments()

# 設定 main.py 專用的 logger，不影響其他模組
logger = logging.getLogger('discord_bot')
# 根據命令行參數設定日誌級別
log_level = logging.DEBUG if args.debug else logging.INFO
logger.setLevel(log_level)

if args.debug:
    logger.debug("除錯模式已啟用")

# 創建格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 創建文件處理器
file_handler = logging.FileHandler('bot.log', encoding='utf-8')
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)

# 創建控制台處理器
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)

# 將處理器添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 防止日誌向上傳播到根 logger
logger.propagate = False

# # 設定 discord.py 的日誌級別（可選：降低 discord.py 的日誌輸出）
# discord_logger = logging.getLogger('discord')
# discord_logger.setLevel(logging.WARNING)  # 只顯示警告和錯誤

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL")
SYSTEM_INSTRUCTION_DIR = os.getenv("SYSTEM_INSTRUCTION_DIR")
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
    logger.error("錯誤：請在 .env 檔案中設定 DISCORD_BOT_TOKEN、 GEMINI_API_KEY 和 MODEL")
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
    logger.debug(f"新增歷史紀錄: user_id={user_id}, role={role}, token_count={token_count}")
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
    """從指定目錄載入所有檔案內容到系統提示"""
    global SYSTEM_PROMPT
    try:
        if not SYSTEM_INSTRUCTION_DIR or not os.path.exists(SYSTEM_INSTRUCTION_DIR):
            logger.warning(f"警告: {SYSTEM_INSTRUCTION_DIR} 目錄不存在。")
            SYSTEM_PROMPT = ""
            return

        combined_content = []
        # 讀取目錄中的所有檔案
        for filename in sorted(os.listdir(SYSTEM_INSTRUCTION_DIR)):
            file_path = os.path.join(SYSTEM_INSTRUCTION_DIR, filename)
            # 只處理檔案，跳過子目錄
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # 只添加非空內容
                            combined_content.append(f"\n\n{content}")
                            # 顯示檔名和前50個字
                            logger.info(f"讀取檔案: {filename}")
                            logger.info(f"檔案內容前50字: {content[:50]}...")
                except Exception as e:
                    logger.warning(f"警告: 無法讀取檔案 {file_path}: {e}")

        SYSTEM_PROMPT = "\n\n".join(combined_content)
        logger.debug(f"成功載入SYSTEM_PROMPT from {len(combined_content)} 個檔案: {SYSTEM_PROMPT}")
    except Exception as e:
        logger.error(f"載入SYSTEM_PROMPT時發生錯誤: {e}")
        SYSTEM_PROMPT = ""


class FileWatcher():
    def __enter__(self) -> None:
        event_handler = self.FileChangeHandler()
        self.observer = Observer()
        # 監控指定的目錄
        if SYSTEM_INSTRUCTION_DIR and os.path.exists(SYSTEM_INSTRUCTION_DIR):
            self.observer.schedule(event_handler, path=SYSTEM_INSTRUCTION_DIR, recursive=False)
        self.observer.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.observer.stop()
        self.observer.join()

    # 檔案監控事件處理器
    class FileChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            # 當目錄中的任何檔案被修改時重新載入
            if not event.is_directory:
                logger.info(f"檔案 {event.src_path} 已修改，重新載入SYSTEM_PROMPT...")
                load_system_prompt()

        def on_created(self, event):
            # 當目錄中新增檔案時重新載入
            if not event.is_directory:
                logger.info(f"檔案 {event.src_path} 已新增，重新載入SYSTEM_PROMPT...")
                load_system_prompt()

        def on_deleted(self, event):
            # 當目錄中刪除檔案時重新載入
            if not event.is_directory:
                logger.info(f"檔案 {event.src_path} 已刪除，重新載入SYSTEM_PROMPT...")
                load_system_prompt()


# --- Bot 事件監聽 ---
@bot.event
async def on_ready():
    """Bot 啟動時執行的動作"""
    setup_database()
    load_system_prompt()
    logger.info(f'Bot 已登入為 {bot.user}')
    logger.info('------')

# --- Bot 指令 ---
@bot.command(name='bot', aliases=ALIASES)
async def chat_command(ctx: commands.Context, *, message: str):
    """與 Gemini AI 進行對話"""
    user_id = ctx.author.id
    logger.debug(f"收到來自用戶 {ctx.author.name} (ID: {user_id}) 的訊息: {message[:100]}...")

    async with ctx.typing():
        try:
            # 取得歷史紀錄
            history = get_user_history(user_id)
            logger.debug(f"載入了 {len(history)} 筆歷史紀錄")

            history.append(
                types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=message)]
                ),
            )
            config.system_instruction = SYSTEM_PROMPT
            logger.debug(f"使用模型: {MODEL}, 系統提示長度: {len(SYSTEM_PROMPT)}")

            response = genai_client.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
            logger.debug(f"API 回應:\n{response}")
            logger.info(f"生成回應，token 使用量: {response.usage_metadata.total_token_count}")

            # 儲存對話紀錄
            # 使用者訊息
            add_history(user_id, 'user', message)
            # 模型回應
            # response.usage_metadata.total_token_count 包含了提示和回應的 token 總數
            token_count = response.usage_metadata.total_token_count
            add_history(user_id, 'model', response.text, token_count)

            # 回傳結果到 Discord
            chunks = split_string_by_length_and_newline(response.text, max_len=2000)
            logger.debug(f"回應被分割成 {len(chunks)} 個片段")
            for i, chunk in enumerate(chunks):
                logger.debug(f"發送片段 {i+1}/{len(chunks)}")
                await ctx.reply(chunk)

        except ServerError as e:
            logger.error(f"ServerError：{e}")
            await ctx.reply(f"ServerError：{e}")
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.error(traceback.format_exc())
            await ctx.reply(f"糟糕，發生錯誤了：{e}")

@bot.command(name='update_system_prompt', aliases=['up'])
async def update_system_prompt_command(ctx: commands.Context):
    """更新 SYSTEM_PROMPT，從訊息的附件下載新的系統指令檔案"""
    if not SYSTEM_INSTRUCTION_DIR:
        await ctx.reply("錯誤：未設定 SYSTEM_INSTRUCTION_DIR 環境變數")
        return

    if not ctx.message.attachments:
        await ctx.reply("錯誤：請在訊息中附加要更新的系統指令檔案")
        return

    logger.info(f"用戶 {ctx.author.name} 請求更新 SYSTEM_PROMPT，附件數量: {len(ctx.message.attachments)}")

    async with ctx.typing():
        temp_backup_dir = None
        downloaded_files = []

        try:
            # 創建暫存目錄並備份原有檔案
            temp_backup_dir = tempfile.mkdtemp(prefix="system_prompt_backup_")
            logger.info(f"創建暫存備份目錄: {temp_backup_dir}")

            os.makedirs(SYSTEM_INSTRUCTION_DIR, exist_ok=True)

            # 備份原有檔案
            for filename in os.listdir(SYSTEM_INSTRUCTION_DIR):
                src_path = os.path.join(SYSTEM_INSTRUCTION_DIR, filename)
                if os.path.isfile(src_path):
                    dst_path = os.path.join(temp_backup_dir, filename)
                    shutil.move(src_path, dst_path)
                    logger.debug(f"備份檔案: {filename}")

            # 下載附件到 SYSTEM_INSTRUCTION_DIR
            async with aiohttp.ClientSession() as session:
                for attachment in ctx.message.attachments:
                    file_path = os.path.join(SYSTEM_INSTRUCTION_DIR, attachment.filename)
                    logger.info(f"下載附件: {attachment.filename}")

                    try:
                        async with session.get(attachment.url) as response:
                            if response.status == 200:
                                content = await response.read()
                                with open(file_path, 'wb') as f:
                                    f.write(content)
                                downloaded_files.append(file_path)
                                logger.info(f"成功下載: {attachment.filename} ({len(content)} bytes)")
                            else:
                                raise Exception(f"下載失敗，HTTP 狀態碼: {response.status}")
                    except Exception as e:
                        logger.error(f"下載附件 {attachment.filename} 失敗: {e}")
                        raise

            load_system_prompt()
            if SYSTEM_PROMPT == "":
                logger.error("載入 SYSTEM_PROMPT 後發現內容為空，可能讀取檔案有問題")
                raise Exception("載入 SYSTEM_PROMPT 失敗，內容為空")

            shutil.rmtree(temp_backup_dir)
            logger.info("SYSTEM_PROMPT 更新成功，暫存目錄已清理")

            await ctx.reply(f"✅ SYSTEM_PROMPT 更新成功！\n已下載 {len(downloaded_files)} 個檔案：\n" +
                          "\n".join([f"• {os.path.basename(f)}" for f in downloaded_files]))

        except Exception as e:
            logger.error(f"更新 SYSTEM_PROMPT 失敗: {e}")

            # 清理下載的檔案並還原備份
            try:
                # 刪除下載的檔案
                for file_path in downloaded_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"清理下載檔案: {os.path.basename(file_path)}")

                # 還原備份檔案
                if temp_backup_dir and os.path.exists(temp_backup_dir):
                    for filename in os.listdir(temp_backup_dir):
                        src_path = os.path.join(temp_backup_dir, filename)
                        dst_path = os.path.join(SYSTEM_INSTRUCTION_DIR, filename)
                        shutil.move(src_path, dst_path)
                        logger.debug(f"還原備份檔案: {filename}")

                    # 清理暫存目錄
                    shutil.rmtree(temp_backup_dir)
                    logger.info("已還原備份檔案並清理暫存目錄")

                # 重新載入原有的 SYSTEM_PROMPT
                load_system_prompt()

            except Exception as cleanup_error:
                logger.error(f"清理和還原過程中發生錯誤: {cleanup_error}")

            await ctx.reply(f"❌ SYSTEM_PROMPT 更新失敗：{e}\n已還原到原始狀態")

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
    if SYSTEM_INSTRUCTION_DIR:
        # with FileWatcher():
        bot.run(DISCORD_BOT_TOKEN)
    else:
        bot.run(DISCORD_BOT_TOKEN)
