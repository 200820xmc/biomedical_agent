"""
批量 PDF → Markdown 转换脚本
使用 markitdown 将文献目录下所有 PDF 转为 MD
"""
import os
import subprocess
import time

# === 配置 ===
input_root = r"C:\Users\Ming\Desktop\动静脉瘘状态监测\文献"
output_root = r"C:\Users\Ming\Desktop\动静脉瘘状态监测\文献_md"

os.makedirs(output_root, exist_ok=True)

total = 0
success = 0
fail = 0
failed_list = []

# 遍历所有子目录
for dirpath, dirnames, filenames in os.walk(input_root):
    for fname in filenames:
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(dirpath, fname)

        # 保持子目录结构
        rel_dir = os.path.relpath(dirpath, input_root)
        out_subdir = os.path.join(output_root, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        # 输出文件名：PDF文件名 → MD文件名
        md_name = os.path.splitext(fname)[0] + ".md"
        md_path = os.path.join(out_subdir, md_name)

        total += 1
        safe_name = fname.encode('ascii', errors='replace').decode('ascii')
        print(f"[{total}/30] {safe_name[:60]}...")

        try:
            result = subprocess.run(
                ["markitdown", pdf_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0 and result.stdout.strip():
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                success += 1
                size_kb = len(result.stdout) / 1024
                print(f"    OK ({size_kb:.1f} KB)")
            else:
                fail += 1
                failed_list.append(fname)
                print(f"    FAIL: {result.stderr[:100] if result.stderr else 'empty output'}")

        except subprocess.TimeoutExpired:
            fail += 1
            failed_list.append(fname)
            print(f"    TIMEOUT (>120s)")
        except Exception as e:
            fail += 1
            failed_list.append(fname)
            print(f"    ERROR: {e}")

        time.sleep(1)

print(f"\n{'='*50}")
print(f"完成！总计 {total} 篇 | 成功 {success} 篇 | 失败 {fail} 篇")
print(f"输出目录: {output_root}")
if failed_list:
    print(f"\n失败文件:")
    for f in failed_list:
        print(f"  - {f}")
