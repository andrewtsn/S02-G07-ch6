"""วัดว่ารุ่น Gemini ที่บัญชีนี้ใช้ได้ รุ่นไหนตอบเร็วและตอบสำเร็จ

วิธีใช้ ในโฟลเดอร์ S02-G07_ui ที่เปิด venv และ export คีย์แล้ว
    python3 S02-G07_modelcheck.py
สคริปต์นี้ไม่พิมพ์คีย์ออกมา และใช้คำถามสั้นมาก ข้อละไม่กี่โทเคน
"""
import os
import time

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=15000))

names = []
for m in client.models.list():
    name = m.name.split("/")[-1]
    if "generateContent" in (m.supported_actions or []) and "flash" in name and not any(
        x in name for x in ("image", "tts", "audio", "live", "embedding", "exp")
    ):
        names.append(name)
print("รุ่น flash ที่บัญชีนี้ใช้ได้", len(names), "รุ่น")
for n in names:
    print("  ", n)
print()

PROMPT = "ตอบสั้นๆ หนึ่งประโยค ปาล์มน้ำมันคือพืชอะไร"
for name in names[:10]:
    results = []
    for _ in range(3):
        t = time.perf_counter()
        try:
            r = client.models.generate_content(model=name, contents=PROMPT)
            ok = bool((r.text or "").strip())
            results.append(f"{round((time.perf_counter() - t) * 1000)} ms" if ok else "ว่าง")
        except Exception as exc:
            results.append(f"ล้ม {getattr(exc, 'code', None) or type(exc).__name__}")
    good = sum(x.endswith("ms") for x in results)
    print(f"{name:40} สำเร็จ {good} จาก 3   " + "   ".join(results))
