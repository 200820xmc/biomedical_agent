"""
批量上传 MD 到 AVF 科研助手知识库
"""
import requests
import os
import time

md_root = r"C:\Users\Ming\Desktop\动静脉瘘状态监测\文献_md"
upload_url = "http://localhost:9900/api/upload"

total = 0
success = 0

for dirpath, dirnames, filenames in os.walk(md_root):
    for fname in filenames:
        if not fname.endswith(".md"):
            continue

        total += 1
        filepath = os.path.join(dirpath, fname)
        safe_name = fname.encode('ascii', errors='replace').decode('ascii')

        print(f"[{total}] {safe_name[:55]}...")
        try:
            with open(filepath, "rb") as f:
                resp = requests.post(upload_url, files={"file": (fname, f)}, timeout=120)
            data = resp.json()
            if data.get("code") == 200:
                success += 1
                print(f"    OK ({data['data']['size']} bytes)")
            else:
                print(f"    FAIL: {data.get('message', 'unknown')}")
        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(0.5)

print(f"\nDone: {success}/{total} uploaded")
