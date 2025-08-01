import os
import discord
from google import genai
from dotenv import load_dotenv
from google.genai import types


character = None
with open("character.txt", "r", encoding="utf8") as f:
    character = f.read()

# 載入.env 檔案中的環境變數
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
if not TOKEN or not GEMINI_KEY:
    exit()

# 設定 Gemini API
genai_client = genai.Client(api_key=GEMINI_KEY)

# 設定意圖 (Intents)
intents = discord.Intents.default()
intents.message_content = True  # 啟用訊息內容意圖
intents.messages = True

# 建立 Bot 客戶端
discord_client = discord.Client(intents=intents)

# 當 Bot 準備就緒時觸發的事件
@discord_client.event
async def on_ready():
    print(f'成功登入為 {discord_client.user}！')

# 當收到訊息時觸發的事件
@discord_client.event
async def on_message(message):
    global character
    # 忽略來自 Bot 自己的訊息，防止無限循環
    if message.author == discord_client.user:
        return

    # 檢查是否有附件
    if message.attachments:
        for attachment in message.attachments:
            # 檢查檔案類型是否是 txt
            if attachment.filename == ('character.txt'):
                file_content = await attachment.read()  # 讀取 bytes
                try:
                    # 嘗試將 bytes 轉為文字（假設 UTF-8 編碼）
                    text = file_content.decode('utf-8')
                    if text:
                        character = text
                        print(character)
                        await message.reply("已更新角色設定")
                    else:
                        await message.reply("略過空白檔案")
                except UnicodeDecodeError:
                    await message.channel.send("無法解碼此 txt 檔案，請確認是否為 UTF-8 格式")
                    return
        return

    # 檢查訊息是否以 '!bot' 開頭
    if message.content.startswith('!bot '):
        # 取得 '!bot ' 後面的問題內容
        prompt = message.content[5:]
        try:
            # 顯示 "Bot 正在輸入..."
            async with message.channel.typing():
                # 將問題發送給 Gemini API
                response = genai_client.models.generate_content(
                    model='gemini-2.0-flash-lite-001',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=
                        [
                            character,
                        ]
                    ),
                )

                # 將 AI 的回覆傳送回 Discord 頻道
                chunks = split_string_by_length_and_newline(response.text, max_len=2000)
                for chunk in chunks:
                    await message.reply(chunk)
        except Exception as e:
            await message.reply(f"糟糕，發生錯誤了：{e}")


def split_string_by_length_and_newline(s, max_len):
    result = []
    start = 0

    while start < len(s):
        # 嘗試切到 max_len 範圍內最後一個 \n 的位置
        end = start + max_len
        if end >= len(s):
            end = len(s)

        sub = s[start:end]
        last_newline = sub.rfind('\n')

        if last_newline == -1:
            # 改為強制切斷，不保證以 \n 結尾
            last_newline = len(sub) - 1  # 或者找其他替代策略

        chunk = s[start:start + last_newline + 1]
        result.append(chunk)
        start += last_newline + 1

    return result

# 使用您的 Token 執行 Bot
discord_client.run(TOKEN)
