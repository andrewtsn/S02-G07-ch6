"""ทดสอบถามตอบจริงด้วยโมเดล MiniLM ตัวจริง และ Gemini จริงถ้าตั้งค่าคีย์ไว้

วิธีใช้ ในโฟลเดอร์ S02-G07_ui ที่เปิด venv แล้ว
    python3 S02-G07_selftest.py
สคริปต์นี้ไม่พิมพ์คีย์ออกมา แคปผลทั้งหมดส่งกลับได้เลย
"""
import os
import time

os.environ.pop("S02G07_TEST_EMBEDDER", None)
t0 = time.perf_counter()
import app  # noqa: E402  loads the real MiniLM and the 41 triples

print(f"โหลดโมเดลและข้อมูล {time.perf_counter() - t0:.1f} วินาที")
print("โมเดล", app.EMBEDDER.name, "อุปกรณ์", getattr(getattr(app.EMBEDDER, "m", None), "device", "ไม่ทราบ"))
print("Gemini", "ตั้งค่าคีย์แล้ว รุ่น " + app.GEMINI_MODEL if app.gemini_ready() else "ยังไม่ได้ตั้งค่าคีย์ จะข้ามการทดสอบ Gemini")
print()

# expected rows come from data/triples.csv
CASES = [
    ("การปลูกปาล์มน้ำมันภาคใต้ต้องใช้น้ำชลประทานเท่าไร", {"T30"}),
    ("ขั้นการสกัดน้ำมันปาล์มดิบปล่อยก๊าซเรือนกระจกเท่าไร", {"T36"}),
    ("การปลูกปาล์มน้ำมันค่าเฉลี่ยประเทศใช้ปุ๋ย N เท่าไร", {"T17", "T22"}),
    ("การสกัดน้ำมันปาล์มดิบชุดหลักได้เส้นใยเท่าไร", {"T25"}),
    ("รถไฟฟ้าสายสีม่วงเที่ยวสุดท้ายออกกี่โมง", None),
    ("การขนส่งผลปาล์มทะลายสดจากสวนไปโรงสกัดมีระยะทางขนส่งไปกลับเท่าไร", {"T14"}),
    ("โนบิตะผลิตปาล์มน้ำมันได้เท่าไหร่", None),
    ("โดเรม่อนใช้ปุ๋ยเท่าไร", None),
    ("ราคาปาล์มวันนี้เท่าไร", None),
]

passed = 0
times = []
for q, expect in CASES:
    r, _ = app.search_result(q)
    times.append(r["search_ms"])
    used = {r["hits"][i - 1]["source_id"] for i in r["used_ranks"]}
    if expect is None:
        ok = r["mode"] in ("not_found", "check")
    else:
        ok = r["mode"] == "template" and bool(used & expect)
    passed += ok
    top3 = "  ".join(f"{h['source_id']} {h['similarity']:.4f}" for h in r["hits"][:3])
    print("ผ่าน" if ok else "ต้องดู", "|", q)
    print("   ค้นหา", r["search_ms"], "มิลลิวินาที  สามอันดับแรก", top3)
    print("   คำตอบ", r["answer"], "| คำที่ไม่รู้จัก", " ".join(app.unknown_terms(q)) or "ไม่มี")
    print("   คาดหวัง", "ไม่พบข้อมูล" if expect is None else " หรือ ".join(sorted(expect)))
    print()

print(f"โหมดนำเสนอ ผ่าน {passed} จาก {len(CASES)} ข้อ  เวลาค้นหาเฉลี่ย {sum(times) / len(times):.0f} มิลลิวินาที")

if app.gemini_ready():
    print("\nทดสอบ Gemini จริง รอสูงสุดข้อละ", app.GEMINI_TIMEOUT_S, "วินาที")
    g_ok = 0
    for q, expect in CASES[:4] + CASES[6:7]:
        g = app.gemini_result(q)
        g_ok += bool(g.get("ok"))
        print("สำเร็จ" if g.get("ok") else "ไม่สำเร็จ", "|", q, "|", g.get("generate_ms", 0), "มิลลิวินาที")
        print("   ", g.get("answer") or g.get("notice"))
    print(f"Gemini สำเร็จ {g_ok} จาก 5 ข้อ ข้อสุดท้ายคือข้อโนบิตะ ที่ถูกคือ Gemini ตอบว่าไม่พบข้อมูลที่เกี่ยวข้อง")
    print("ใช้ thinking ระดับต่ำได้" if app._gemini.get("low_thinking") else "รุ่นนี้ไม่รับ thinking ระดับต่ำ ระบบถามแบบปกติแทนแล้ว")
