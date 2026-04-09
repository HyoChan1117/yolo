import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SLACK_WEBHOOK_URL", "")
if not url:
    print("❌ .env에 SLACK_WEBHOOK_URL이 없어요.")
    exit(1)

for msg in ["🔓 문 열었습니다.", "🔒 문 닫았습니다."]:
    res = requests.post(url, json={"text": msg}, timeout=5)
    if res.status_code == 200:
        print(f"✅ 전송 성공: {msg}")
    else:
        print(f"❌ 전송 실패 ({res.status_code}): {res.text}")
