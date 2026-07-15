"""临时脚本：交叉验证 questions.csv 中的文件引用是否在 uploads/ 中存在。"""
import csv, os, sys

sys.path.insert(0, ".")
# Force fresh import
for k in list(sys.modules):
    if "evaluation" in k:
        del sys.modules[k]
from evaluation.common import normalize_file_name

# ── 读取问题集 ──
rows = []
with open("evaluation/questions.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"Total questions: {len(rows)}")

# ── 获取 uploads 实际文件 ──
actual_files = sorted([f for f in os.listdir("uploads") if f.endswith((".md", ".txt"))])
actual_norm = {normalize_file_name(f): f for f in actual_files}
print(f"Uploads files: {len(actual_files)}")
print()

# ── 逐题检查 ──
total_refs = 0
matched = 0
missing = []
needs_review = []

for q in rows:
    qid = q.get("question_id", "?")
    relevant = q.get("relevant_files", "").strip()
    category = q.get("category", "?")

    if not relevant or relevant.upper() == "NEEDS_REVIEW":
        needs_review.append(qid)
        continue

    refs = [r.strip() for r in relevant.split(";") if r.strip()]
    total_refs += len(refs)

    for ref in refs:
        # Exact match
        if ref in actual_files:
            matched += 1
            continue

        # Normalized match
        ref_norm = normalize_file_name(ref)
        found = actual_norm.get(ref_norm)
        if found:
            matched += 1
            if ref != found:
                missing.append(f"[{qid}] FUZZY: ref -> actual: {ref[:60]}... -> {found[:60]}...")
            continue

        # Try harder: word-level overlap for encoding-corrupted refs
        ref_words = set(ref_norm.split())
        best_match = None
        best_jaccard = 0.0
        for af_norm, af in actual_norm.items():
            af_words = set(af_norm.split())
            common = ref_words & af_words
            if not common:
                continue
            jaccard = len(common) / len(ref_words | af_words)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = af

        if best_match and best_jaccard > 0.5:
            matched += 1
            missing.append(f"[{qid}] FUZZY-WORD ({best_jaccard:.2f}): ref -> actual: {ref[:50]}... -> {best_match[:50]}...")
            continue

        missing.append(f"[{qid}] MISSING (best jaccard={best_jaccard:.2f}): {ref[:60]}...")

# ── 汇总 ──
print("=== Cross-Reference Results ===")
print(f"Total references: {total_refs}")
print(f"Matched: {matched}")
print(f"Missing: {len(missing)}")
if total_refs:
    print(f"Match rate: {matched}/{total_refs} = {matched/total_refs*100:.1f}%")
print()

if needs_review:
    print(f"NEEDS_REVIEW questions: {len(needs_review)} ({', '.join(needs_review)})")
    print()

if missing:
    print(f"=== Issues ({len(missing)}) ===")
    for m in missing:
        print(m)
else:
    print("ALL REFERENCES VALID!")

# Song duplicate check
print()
song_files = [f for f in actual_files if "song" in f.lower() and "2023" in f]
if len(song_files) > 1:
    print(f"WARNING: {len(song_files)} Song 2023 files found:")
    for sf in song_files:
        print(f"  - {sf}")
