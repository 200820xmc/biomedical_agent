"""RAGAS 风格评测 — 使用 LLM 评判 RAG 四项核心指标

指标（参考 RAGAS 论文）：
- Faithfulness：答案中的每个陈述是否都能在上下文中找到依据
- Answer Relevancy：答案是否直接回应了问题
- Context Precision：检索到的上下文是否相关（precision@k）
- Context Recall：上下文覆盖了答案中的关键信息

用法：
    python evaluation/evaluate_ragas.py --questions evaluation/questions_v2.csv --limit 10
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ChatQwen 需要 DASHSCOPE_API_BASE 指向中国站
import os as _os
_os.environ.setdefault(
    "DASHSCOPE_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

from loguru import logger
from langchain_openai import ChatOpenAI

from app.config import config
from app.services.retrieval import retrieval_service
from evaluation.common import load_questions, write_json, write_csv

CST = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════════════
# RAGAS 风格评判 Prompt
# ═══════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """你的任务是判断"答案"中的每个陈述是否都能在"上下文"中找到依据。

问题：{question}
上下文：{context}
答案：{answer}

请判断答案的忠实度（0~1分）：
- 1.0：答案中的所有陈述都能在上下文中找到准确依据
- 0.7~0.9：绝大部分陈述有依据，少数细节略有偏差
- 0.4~0.6：部分陈述有依据，部分为推测
- 0.1~0.3：仅少量陈述有依据，大部分为推测
- 0.0：答案与上下文完全无关或全部为编造

请只输出一个 0~1 的数字表示忠实度分数，不要输出其他内容。"""

ANSWER_RELEVANCY_PROMPT = """你的任务是判断"答案"是否直接回应了"问题"，以及回答的完整程度。

问题：{question}
答案：{answer}

请判断答案的相关度和完整度（0~1分）：
- 1.0：答案完全回应问题，信息完整且直接
- 0.7~0.9：答案基本回应问题，但缺少部分细节
- 0.4~0.6：答案部分回应问题，但偏离了核心
- 0.1~0.3：答案与问题仅有微弱关联
- 0.0：答案与问题完全无关

请只输出一个 0~1 的数字表示相关度分数，不要输出其他内容。"""

CONTEXT_PRECISION_PROMPT = """你的任务是判断"上下文"中的每条内容是否与"问题"相关。

问题：{question}
上下文：{context}

请判断上下文的精度（0~1分）——即上下文中有多少是真正回答该问题所需的：
- 1.0：所有上下文内容都直接相关，没有任何冗余
- 0.7~0.9：大部分上下文相关，少量冗余
- 0.4~0.6：约一半相关，一半无关
- 0.1~0.3：大部分上下文与问题无关
- 0.0：所有上下文都与问题无关

请只输出一个 0~1 的数字表示精度分数，不要输出其他内容。"""

CONTEXT_RECALL_PROMPT = """你的任务是判断回答问题所需的关键信息是否都在"上下文"中被覆盖到了。

问题：{question}
答案：{answer}
上下文：{context}

请判断上下文的召回率（0~1分）——即答案中的关键信息有多少能在上下文中找到：
- 1.0：答案中的所有关键信息都能在上下文中找到
- 0.7~0.9：大部分关键信息能找到，少数缺失
- 0.4~0.6：约一半关键信息能找到
- 0.1~0.3：仅少量关键信息能找到
- 0.0：答案关键信息完全不在上下文中

请只输出一个 0~1 的数字表示召回分数，不要输出其他内容。"""


def _build_eval_llm() -> ChatOpenAI:
    """构建评测专用 LLM（temperature=0，稳定打分）"""
    return ChatOpenAI(
        model=config.dashscope_model,
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=config.dashscope_api_key,
    )


async def _score(llm: ChatOpenAI, prompt: str) -> float | None:
    """调用 LLM 打分，返回 0~1 的浮点数"""
    try:
        import asyncio as _asyncio
        response = await _asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=30.0,
        )
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        # 提取首个数字
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        logger.warning(f"LLM 打分失败: {e}")
        return None


def _limit_text(text: str, max_chars: int = 3000) -> str:
    """截断文本以防止 prompt 过长"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(已截断)"


async def run_ragas_eval(
    questions_csv: str,
    limit: int = 0,
    output_dir: str = "",
) -> dict[str, Any]:
    questions, errors = load_questions(questions_csv)
    if limit > 0:
        questions = questions[:limit]

    logger.info(f"加载 {len(questions)} 道评测题目")

    eval_llm = _build_eval_llm()

    details: list[dict] = []
    all_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i, q in enumerate(questions):
        question = q["question"]
        qid = q["question_id"]
        logger.info(f"[{i+1}/{len(questions)}] {qid}")

        try:
            # 1. 检索
            context, artifact_dict = await retrieval_service.retrieve(query=question, search_mode="auto")
            ctx_text = _limit_text(context, 3000)

            # 2. 获取答案（非流式）
            from app.services.rag_agent_service import rag_agent_service
            answer = await rag_agent_service.query(question=question, session_id=f"ragas-{qid}")
            answer = answer[:2000] if answer else "（无回答）"

            # 3. 四项评判
            scores = {}
            for metric, prompt_tpl in [
                ("faithfulness", FAITHFULNESS_PROMPT),
                ("answer_relevancy", ANSWER_RELEVANCY_PROMPT),
                ("context_precision", CONTEXT_PRECISION_PROMPT),
                ("context_recall", CONTEXT_RECALL_PROMPT),
            ]:
                prompt = prompt_tpl.format(question=question, context=ctx_text, answer=answer)
                score = await _score(eval_llm, prompt)
                if score is not None:
                    scores[metric] = round(score, 4)
                    all_scores[metric].append(score)

            details.append({
                "question_id": qid,
                "question": question[:200],
                "answer_preview": answer[:300],
                "context_chars": len(context),
                **scores,
            })
            logger.info(f"  scores={scores}")

        except Exception as e:
            logger.error(f"  {qid} 失败: {e}")
            details.append({"question_id": qid, "question": question[:100], "error": str(e)})

        # 避免 API 限流
        await asyncio.sleep(0.5)

    # 汇总
    metrics_summary = {}
    for metric, values in all_scores.items():
        if values:
            metrics_summary[metric] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }

    run_id = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "ran_at": datetime.now(CST).isoformat(),
        "question_file": questions_csv,
        "question_count": len(questions),
        "framework": "RAGAS-style (custom LLM judge)",
        "metrics": metrics_summary,
    }

    if not output_dir:
        output_dir = str(Path(__file__).resolve().parent / "results" / run_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(summary, str(Path(output_dir) / "ragas_summary.json"))
    write_csv(details, str(Path(output_dir) / "ragas_details.csv"))

    print("\n" + "=" * 60)
    print("RAGAS-style 评测结果")
    print("=" * 60)
    for metric, info in metrics_summary.items():
        print(f"  {metric:25s}: {info['mean']:.4f}  (n={info['count']})")
    print(f"\n结果保存至: {output_dir}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAGAS-style RAG 质量评测")
    parser.add_argument("--questions", default="evaluation/questions_v2.csv")
    parser.add_argument("--limit", type=int, default=5, help="题目数（0=全部）")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    asyncio.run(run_ragas_eval(args.questions, args.limit, args.output))


if __name__ == "__main__":
    main()
