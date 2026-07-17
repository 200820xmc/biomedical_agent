"""生成证据先行的 50 题 Ragas 评测集。

安全原则：
1. 先选择文献证据，再编写问题，避免问题与语料脱节。
2. 每题只绑定一个可追踪的参考证据块，不把整篇文档当作相关上下文。
3. 不自动生成 reference，防止未经审核的模型答案成为错误真值。
4. 通过 SHA-256 固定证据版本，并生成单独的人工复核清单。

输出：
- evaluation/ragas_50_dataset.jsonl：可转换为 Ragas SingleTurnSample 的核心字段。
- evaluation/ragas_50_manifest.jsonl：题目、证据 ID 和来源元数据。
- evaluation/ragas_50_review.csv：人工审核表。
- evaluation/ragas_50_summary.json：生成与校验摘要。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARSED_ROOT = ROOT / "uploads" / "parsed"
OUTPUT_DIR = ROOT / "evaluation"

DATASET_PATH = OUTPUT_DIR / "ragas_50_dataset.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "ragas_50_manifest.jsonl"
REVIEW_PATH = OUTPUT_DIR / "ragas_50_review.csv"
SUMMARY_PATH = OUTPUT_DIR / "ragas_50_summary.json"

GENERATION_VERSION = "evidence_first_extractive_v1"
CHUNKING_VERSION = "xparse_reference_context_v1"
MAX_CONTEXT_CHARS = 2400
MIN_CONTEXT_CHARS = 180


@dataclass(frozen=True)
class QuestionSpec:
    document_id: str
    question: str
    category: str
    question_type: str


QUESTION_SPECS = [
    QuestionSpec("doc_0c0251", "吻合口角度如何影响侧端动静脉瘘中的扰动流分布？", "血流动力学与几何", "mechanism"),
    QuestionSpec("doc_0cfc23", "计算机化血管声音分析发现哪些声学变化与透析通路狭窄程度相关？", "声学监测", "extractive"),
    QuestionSpec("doc_132164", "低频声音在人体胸腔组织传播时，肺共振与骨骼的相对影响如何？", "声传播机制", "comparison"),
    QuestionSpec("doc_16308d", "Song 等提出的动静脉瘘狭窄 AI 模型使用了哪些输入特征和分类器？", "AI与信号处理", "extractive"),
    QuestionSpec("doc_190ed9", "小波变换系数和支持向量机如何用于动静脉瘘狭窄分类？", "AI与信号处理", "method"),
    QuestionSpec("doc_26b3f0", "皮肤耦合 PVDF 多通道传感器如何用于非侵入性血管通路监测？", "声学监测", "method"),
    QuestionSpec("doc_29cc5e", "定量血管声学分析能够估计哪些动脉血流和几何参数？", "声学监测", "extractive"),
    QuestionSpec("doc_30e955", "子带自回归线性预测如何增强 PAG 信号并辅助评估再狭窄风险？", "AI与信号处理", "method"),
    QuestionSpec("doc_3501e6", "用于研究动脉杂音生物力学的耦合计算框架包含哪些主要模型？", "计算声学", "extractive"),
    QuestionSpec("doc_3899ee", "Park 等使用哪些深度卷积神经网络分析动静脉瘘听诊数据，其主要结果是什么？", "AI与信号处理", "method_and_result"),
    QuestionSpec("doc_3b7890", "KDOQI 2019 血管通路指南如何体现以患者 ESKD Life-Plan 为核心的管理理念？", "临床指南", "explanation"),
    QuestionSpec("doc_3e2e29", "纵向 CFD 研究观察到动静脉瘘成熟、重塑和狭窄形成怎样的时间变化？", "血流动力学与几何", "temporal"),
    QuestionSpec("doc_4058a8", "该血管杂音理论如何解释近场、远场以及逆行声波的作用？", "声传播机制", "mechanism"),
    QuestionSpec("doc_42053f", "三种最常见的动静脉瘘分别是什么，其特征性狭窄部位有什么共同几何特征？", "血流动力学与几何", "comparison"),
    QuestionSpec("doc_46b880", "基于听诊特征和支持向量机的动静脉瘘狭窄诊断算法取得了怎样的交叉验证结果？", "AI与信号处理", "quantitative"),
    QuestionSpec("doc_49476e", "PAG 信号中哪些听觉和频谱特征可用于估计狭窄位置和程度？", "声学监测", "extractive"),
    QuestionSpec("doc_57bca3", "动静脉瘘吻合角度综述如何划分角度范围，不同研究分别支持哪些角度？", "血流动力学与几何", "comparison"),
    QuestionSpec("doc_5e5413", "空化流噪声的直接数值模拟采用了什么模型，云空化塌陷如何影响噪声？", "计算声学", "method_and_result"),
    QuestionSpec("doc_61c5d8", "2010 年全球 RRT 实际接受人数、需求缺口和 2030 年预测规模分别是多少？", "流行病学", "quantitative"),
    QuestionSpec("doc_6aca02", "与 MRA 相比，CTA 检测颅内血管狭窄和闭塞的敏感性如何？", "医学影像", "comparison"),
    QuestionSpec("doc_6f1019", "多点声学测量中，显著狭窄前后收缩期频谱内容的起始时间发生了什么变化？", "声学监测", "temporal"),
    QuestionSpec("doc_73baa9", "耦合流声计算研究认为狭窄动脉杂音的主要声源与哪种壁面压力作用有关？", "计算声学", "mechanism"),
    QuestionSpec("doc_75321e", "胸腔血管噪声模型中，声功率与雷诺数及血管直径比具有怎样的标度关系？", "计算声学", "quantitative"),
    QuestionSpec("doc_799901", "ESKD 综述指出哪些人群面临医疗可及性差异，未来护理重点是什么？", "流行病学", "extractive"),
    QuestionSpec("doc_806271", "PCA 和分层 rs-SOM 如何改进血液透析狭窄的人工听诊系统？", "AI与信号处理", "method"),
    QuestionSpec("doc_85adcd", "狭窄引起的湍流如何激励柔性血管壁及周围组织产生振动？", "声传播机制", "mechanism"),
    QuestionSpec("doc_8a162a", "小波变换如何用于动静脉瘘状态的多分类评估，其结果与传统检查有什么关系？", "AI与信号处理", "method_and_result"),
    QuestionSpec("doc_8b08d9", "低且振荡的壁面剪切应力出现在哪些动静脉瘘区域，与狭窄部位有什么关系？", "血流动力学与几何", "mechanism"),
    QuestionSpec("doc_8c4add", "以中心静脉导管开始透析后选择动静脉瘘或人工血管，对手术次数和年度通路费用有何影响？", "卫生经济学", "comparison"),
    QuestionSpec("doc_8ec9d8", "患者特异性动静脉瘘的 FSI 模拟为何需要非牛顿血液模型，简化模拟能达到什么精度？", "血流动力学与几何", "method_and_result"),
    QuestionSpec("doc_91bbb0", "多通道血管杂音传感器在体外狭窄模型中测得了怎样的带宽和动态范围？", "声学监测", "quantitative"),
    QuestionSpec("doc_91d334", "该研究为什么要为 k-NN 进行特征选择，目标分类任务是什么？", "AI与信号处理", "method"),
    QuestionSpec("doc_96d0a0", "ESRD 医学管理中，肾移植、透析和保守治疗分别适用于哪些情况？", "临床管理", "comparison"),
    QuestionSpec("doc_991ca5", "动静脉瘘高频血管壁振动具有哪些频率和幅度特征，主要出现在哪些位置？", "血流动力学与几何", "quantitative"),
    QuestionSpec("doc_997c1e", "局部轴对称狭窄后壁面压力波动的幅值和频谱有哪些主要特征？", "计算声学", "extractive"),
    QuestionSpec("doc_9998a7", "该研究通过哪些标记、铸型和成像技术追踪动静脉瘘局部 WSS 与内膜中层厚度变化？", "血流动力学与几何", "method"),
    QuestionSpec("doc_99bb1e", "基于 S 变换的动静脉瘘狭窄检测方法如何构建时频特征，其初步性能如何？", "AI与信号处理", "method_and_result"),
    QuestionSpec("doc_9c1a7b", "狭窄区域及其下游约 5 厘米处的动静脉瘘杂音在音调特征上有何差异？", "声学监测", "comparison"),
    QuestionSpec("doc_9f299f", "吻合口面积和角度如何影响侧端动静脉瘘的压降与流量分配？", "血流动力学与几何", "mechanism"),
    QuestionSpec("doc_a221f8", "动脉狭窄伪声模型中，血管壁压力谱到皮肤表面压力谱的滤波关系是什么？", "声传播机制", "mechanism"),
    QuestionSpec("doc_a40c57", "动静脉瘘体格检查包括哪三个基本步骤，文献使用哪些金标准检查验证异常发现？", "临床管理", "extractive"),
    QuestionSpec("doc_a770c3", "受限通道脉动流的 DNS 和 LES 揭示了狭窄下游哪些主要流动结构？", "计算流体力学", "extractive"),
    QuestionSpec("doc_ac0415", "临床试验中判定 ESRD 需要考虑哪些核心条件和特殊情况？", "临床定义", "extractive"),
    QuestionSpec("doc_add0a1", "计算机辅助动静脉瘘声音分析研究如何采集和处理血管杂音信号？", "声学监测", "method"),
    QuestionSpec("doc_adffac", "MRI 结合 CFD 如何用于观察动静脉瘘成熟过程中的 WSS 和血管重塑变化？", "血流动力学与几何", "method"),
    QuestionSpec("doc_ae606d", "血管声音可视化系统的 INDEX 与平均流量和阻力指数有什么关系？", "声学监测", "quantitative"),
    QuestionSpec("doc_b2d769", "动静脉通路常见的失败并发症包括哪些类型？", "临床管理", "extractive"),
    QuestionSpec("doc_b95bd6", "系统综述如何总结剪切应力、向外重塑和动静脉瘘内膜增生之间的证据？", "血流动力学与几何", "evidence_synthesis"),
    QuestionSpec("doc_bbd5c3", "CFD 研究如何比较侧侧吻合与端侧吻合的压降、流量和壁面剪切应力？", "血流动力学与几何", "comparison"),
    QuestionSpec("doc_c2a542", "用于非侵入性动静脉瘘状态诊断的原型设备由哪些模块组成，研究纳入了多少患者？", "AI与信号处理", "method_and_result"),
]


# 这些词不是参考答案，而是人工选定的“证据存在性哨兵”。
# 生成时若证据块缺少任一词组，脚本立即失败，避免只截到标题或无关背景。
EVIDENCE_CHECKS = {
    "doc_0c0251": ("anastomosis angle", "disturbed flow", "smaller angle"),
    "doc_0cfc23": ("acoustic amplitude", "energy distribution", "stenosis"),
    "doc_132164": ("lung resonances", "bones", "acoustic waves"),
    "doc_16308d": ("short-time Fourier transform", "sample entropy", "ResNet50"),
    "doc_190ed9": ("wavelets transform coefficients", "support vector machine", "classification"),
    "doc_26b3f0": ("multi-channel", "skin surface", "PVDF"),
    "doc_29cc5e": ("arterial diameter", "flow velocity", "wall pressure fluctuations"),
    "doc_30e955": ("sub-band", "linear prediction", "re-stenosis risk"),
    "doc_3501e6": ("immersed boundary method", "linear elastic wave equation", "decomposition"),
    "doc_3899ee": ("DenseNet201", "EfficientNetB5", "ResNet50"),
    "doc_3b7890": ("ESKD Life-Plan", "individualized", "patient"),
    "doc_3e2e29": ("1.5 years", "stenosis", "disturbed flow"),
    "doc_4058a8": ("near-field", "far field", "retrograde"),
    "doc_42053f": ("radiocephalic", "brachiocephalic", "significant angulation"),
    "doc_46b880": ("support vector machine", "90%", "cross-validation"),
    "doc_49476e": ("spectral centroid", "location", "severity"),
    "doc_57bca3": ("acute", "intermediate", "obtuse"),
    "doc_5e5413": ("cloud cavitation", "direct numerical simulation", "noise"),
    "doc_61c5d8": ("2·618 million", "2·284 million", "5·439 million"),
    "doc_6aca02": ("98%", "100%", "87"),
    "doc_6f1019": ("+22", "-20", "systolic spectral content"),
    "doc_73baa9": ("integrated pressure force", "post-stenotic", "time-derivative"),
    "doc_75321e": ("fourth power", "eighth power", "Reynolds number"),
    "doc_799901": ("Black and Hispanic", "life plan", "team-based"),
    "doc_806271": ("principal component analysis", "rs-SOM", "false-positive"),
    "doc_85adcd": ("turbulent flow", "tube wall", "viscoelastic"),
    "doc_8a162a": ("wavelet transform", "multiclass", "classic diagnostic"),
    "doc_8b08d9": ("low and oscillating", "artery floor", "juxta-anastomotic"),
    "doc_8c4add": ("surgical access procedures", "annual cost", "AVF"),
    "doc_8ec9d8": ("non-Newtonian", "low shear rates", "20%"),
    "doc_91bbb0": ("2.25 kHz", "60.2 dB", "multichannel"),
    "doc_91d334": ("features selection", "multiclass classification", "k-NN"),
    "doc_96d0a0": ("kidney transplantation", "shared decision-making", "palliative"),
    "doc_991ca5": ("200 Hz", "200 μm", "anastomosis floor"),
    "doc_997c1e": ("wall pressure", "low-frequency", "new frequency components"),
    "doc_9998a7": ("radiopaque marker", "casting", "micro-MRI"),
    "doc_99bb1e": ("S-transform", "87.84%", "89.24%"),
    "doc_9c1a7b": ("high-pitch", "5cm downstream", "low-pitch"),
    "doc_9f299f": ("pressure drop", "flow distribution", "58°"),
    "doc_a221f8": ("wall of theartery", "surface of the skin", "ω^{-2}"),
    "doc_a40c57": ("inspection", "palpation", "ultrasound and angiography"),
    "doc_a770c3": ("two shear-layers", "vortex structures", "wall pressure"),
    "doc_ac0415": ("symptomatic uremia", "chronicity", "refused"),
    "doc_add0a1": ("electronic stethoscope", "22 000 Hz", "Fast Fourier Transform"),
    "doc_adffac": ("MR angiography", "MR velocimetry", "WSS"),
    "doc_ae606d": ("mean flow volume", "resistance index", "0.68"),
    "doc_b2d769": ("failure to mature", "stenosis formation", "thrombosis"),
    "doc_b95bd6": ("outward remodelling", "oscillating shear stress", "intimal medial thickening"),
    "doc_bbd5c3": ("pressure drop", "venous outflow", "WSS"),
    "doc_c2a542": ("38 patients", "three sub-modules", "classification"),
}


def _normalize_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    # 仅删除真正的 HTML 标签，不能使用宽泛的 <...>，否则会误删 P<0.001
    # 或角度 <30° 等科研证据。
    text = re.sub(r"</?[A-Za-z][^<>]*>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_section(markdown: str, heading_pattern: str) -> str | None:
    pattern = re.compile(
        rf"(?ims)^\s*#{{1,3}}\s*{heading_pattern}\s*$"
        rf"(.*?)"
        rf"(?=^\s*#{{1,3}}\s+\S|\Z)"
    )
    match = pattern.search(markdown)
    if not match:
        return None
    return _normalize_text(match.group(1))


def _extract_custom_context(markdown: str, document_id: str) -> str | None:
    """修复少数版式异常文献的摘要提取，不进行任何内容生成。"""
    if document_id == "doc_6aca02":
        cleaned = _normalize_text(markdown)
        start = cleaned.find("BACKGROUND AND PURPOSE:")
        end = cleaned.find("Stroke is", start)
        if start >= 0 and end > start:
            return cleaned[start:end]

    if document_id == "doc_b95bd6":
        results = _extract_section(markdown, r"Results")
        conclusion = _extract_section(markdown, r"Conclusion")
        if results and conclusion:
            return f"Results\n{results}\n\nConclusion\n{conclusion}"

    return None


def _extract_reference_context(markdown: str, document_id: str) -> str:
    """从文献中提取一个紧凑、可独立审阅的证据块。"""
    candidates = [
        _extract_custom_context(markdown, document_id),
        _extract_section(markdown, r"abstract"),
        _extract_section(markdown, r"summary"),
        _extract_section(markdown, r"executive\s+summary"),
    ]

    context = next(
        (candidate for candidate in candidates if candidate and len(candidate) >= MIN_CONTEXT_CHARS),
        None,
    )

    if context is None:
        cleaned = _normalize_text(markdown)
        heading_positions = list(re.finditer(r"(?m)^#{1,3}\s+\S.*$", cleaned))
        if heading_positions:
            start = heading_positions[0].start()
            context = cleaned[start : start + MAX_CONTEXT_CHARS]
        else:
            context = cleaned[:MAX_CONTEXT_CHARS]

    # 尽量在段落或句号边界截断，避免制造残缺证据。
    if len(context) > MAX_CONTEXT_CHARS:
        window = context[:MAX_CONTEXT_CHARS]
        cut_candidates = [
            window.rfind("\n\n"),
            window.rfind(". "),
            window.rfind("。"),
        ]
        cut = max(cut_candidates)
        if cut >= MIN_CONTEXT_CHARS:
            context = window[: cut + 1]
        else:
            context = window

    context = _normalize_text(context)
    if len(context) < MIN_CONTEXT_CHARS:
        raise ValueError(f"参考证据过短：{len(context)} chars")
    return context


def _locate_markdown(document_id: str) -> Path:
    document_dir = PARSED_ROOT / document_id
    if not document_dir.is_dir():
        raise FileNotFoundError(f"文档目录不存在：{document_dir}")
    files = sorted(path for path in document_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md")
    if len(files) != 1:
        raise ValueError(f"{document_id} 应有且仅有一个 Markdown，实际为 {len(files)} 个")
    return files[0]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if len(QUESTION_SPECS) != 50:
        raise ValueError(f"题目数量必须为 50，实际为 {len(QUESTION_SPECS)}")

    document_ids = [spec.document_id for spec in QUESTION_SPECS]
    questions = [spec.question for spec in QUESTION_SPECS]
    if len(set(document_ids)) != 50:
        raise ValueError("存在重复 document_id")
    if len(set(questions)) != 50:
        raise ValueError("存在完全重复的问题")

    dataset_rows: list[dict] = []
    manifest_rows: list[dict] = []
    review_rows: list[dict] = []

    for index, spec in enumerate(QUESTION_SPECS, start=1):
        question_id = f"rq{index:03d}"
        markdown_path = _locate_markdown(spec.document_id)
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        context = _extract_reference_context(markdown, spec.document_id)
        missing_terms = [
            term
            for term in EVIDENCE_CHECKS[spec.document_id]
            if term.casefold() not in context.casefold()
        ]
        if missing_terms:
            raise ValueError(
                f"{spec.document_id} 的证据块缺少安全校验词：{missing_terms}"
            )
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        context_id = f"{spec.document_id}:{context_hash[:16]}"
        relative_path = markdown_path.relative_to(ROOT).as_posix()

        # 仅保留 Ragas 单轮样本的核心字段。response/retrieved_contexts 在实际运行时填充；
        # reference 必须在人工审核后才能填充。
        dataset_rows.append(
            {
                "user_input": spec.question,
                "response": None,
                "retrieved_contexts": [],
                "retrieved_context_ids": [],
                "reference": None,
                "reference_contexts": [context],
                "reference_context_ids": [context_id],
            }
        )

        manifest_rows.append(
            {
                "question_id": question_id,
                "user_input": spec.question,
                "category": spec.category,
                "question_type": spec.question_type,
                "expected_tool": "retrieve_knowledge",
                "reference_status": "evidence_only_no_answer",
                "review_status": "pending_human_review",
                "generation_method": GENERATION_VERSION,
                "reference_context_ids": [context_id],
                "reference_context_metadata": [
                    {
                        "document_id": spec.document_id,
                        "markdown_path": relative_path,
                        "markdown_filename": markdown_path.name,
                        "evidence_sha256": context_hash,
                        "char_count": len(context),
                        "chunking_version": CHUNKING_VERSION,
                    }
                ],
            }
        )

        review_rows.append(
            {
                "question_id": question_id,
                "question": spec.question,
                "category": spec.category,
                "question_type": spec.question_type,
                "document_id": spec.document_id,
                "markdown_path": relative_path,
                "context_id": context_id,
                "evidence_chars": len(context),
                "automatic_evidence_check": "pass",
                "evidence_preview": context[:500].replace("\n", " "),
                "question_supported": "",
                "evidence_clean": "",
                "reference_answer_reviewed": "",
                "accept": "",
                "reviewer_notes": "",
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATASET_PATH, dataset_rows)
    _write_jsonl(MANIFEST_PATH, manifest_rows)

    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    category_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    context_lengths: list[int] = []
    for spec, dataset_row in zip(QUESTION_SPECS, dataset_rows, strict=True):
        category_counts[spec.category] = category_counts.get(spec.category, 0) + 1
        type_counts[spec.question_type] = type_counts.get(spec.question_type, 0) + 1
        context_lengths.append(len(dataset_row["reference_contexts"][0]))

    summary = {
        "generation_version": GENERATION_VERSION,
        "question_count": len(dataset_rows),
        "unique_question_count": len(set(questions)),
        "unique_document_count": len(set(document_ids)),
        "reference_answer_count": sum(row["reference"] is not None for row in dataset_rows),
        "pending_human_review_count": len(review_rows),
        "automatic_evidence_check_pass_count": len(review_rows),
        "context_length": {
            "min": min(context_lengths),
            "max": max(context_lengths),
            "mean": round(sum(context_lengths) / len(context_lengths), 2),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(sorted(type_counts.items())),
        "recommended_metrics_before_reference_review": [
            "id_based_context_precision",
            "id_based_context_recall",
            "context_utilization",
            "faithfulness",
            "response_relevancy",
        ],
        "metrics_requiring_reviewed_reference_or_claims": [
            "context_recall",
            "answer_correctness",
        ],
        "files": {
            "dataset": DATASET_PATH.relative_to(ROOT).as_posix(),
            "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "review": REVIEW_PATH.relative_to(ROOT).as_posix(),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
