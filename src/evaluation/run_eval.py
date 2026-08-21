
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import RESULTS_DIR, TESTSET_PATH, TOP_K
from generation.generator import generate_differential
from generation.schemas import DifferentialDiagnosisResponse
from ingestion.records import ChunkRecord, record_from_payload
from retrieval.retriever import search_documents

# judge_all_claims now lives in judge.py and batches every claim of a case
# into a SINGLE llm call (was one call per claim).
from evaluation.judge import ClaimJudgement, judge_all_claims
from evaluation.metrics import CaseScores, aggregate, score_case


def load_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    return cases[:limit] if limit else cases


def run_case(case: dict[str, Any], k: int) -> tuple[CaseScores, dict[str, Any]]:
    query = case["query"]

    retrieved = search_documents(query, top_k=k)
    response, debug = generate_differential(query, top_k=k, return_debug=True)

    judgements = judge_all_claims(response, retrieved) if not response.refused else {}

    scores = score_case(
        case_id=case["case_id"],
        response=response,
        retrieved=retrieved,
        judgements=judgements,
        k=k,
        relevant_chunk_ids=case.get("relevant_chunk_ids", []),
        relevant_sections=case.get("relevant_sections", []),
        expected_behavior=case.get("expected_behavior", "answer"),
    )

    record = {
        "case_id": case["case_id"],
        "case_type": case.get("case_type"),
        "query": query,
        "expected_behavior": case.get("expected_behavior"),
        "refused": response.refused,
        "refusal_reason": response.refusal_reason,
        "confidence": response.confidence,
        "top_score": response.top_score,
        "retrieved": debug.get("retrieved", []),
        "claim_audit": debug.get("claim_audit", {}),
        "response_markdown": response.to_markdown(),
        "scores": {
            "precision_at_k": scores.precision_at_k,
            "citation_accuracy": scores.citation_accuracy,
            "faithfulness": scores.faithfulness,
            "total_claims": scores.total_claims,
            "supported_claims": scores.supported_claims,
            "total_citations": scores.total_citations,
            "correct_citations": scores.correct_citations,
        },
        "failures": [asdict(f) | {"failure_type": f.failure_type.value} for f in scores.failures],
        "judgements": {
            claim: {
                "supported_by_retrieved": j.supported_by_retrieved,
                "supporting_chunk_ids": j.supporting_chunk_ids,
                "citation_verdicts": j.citation_verdicts,
                "reason": j.reason,
                "error": j.error,
            }
            for claim, j in judgements.items()
        },
    }
    return scores, record


def fmt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(TESTSET_PATH))
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    cases = load_cases(Path(args.testset), args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[CaseScores] = []
    all_records: list[dict[str, Any]] = []
    judge_errors = 0

    print(f"Running {len(cases)} cases at k={args.k}...")
    print("Budget: ~1 generation call + ~1 judge call per case.\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['case_id']} — {case['query'][:60]}...")
        try:
            scores, record = run_case(case, args.k)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        errs = sum(1 for j in record["judgements"].values() if j.get("error"))
        judge_errors += errs

        all_scores.append(scores)
        all_records.append(record)
        print(
            f"    P@{args.k}={fmt(scores.precision_at_k)}  "
            f"CitAcc={fmt(scores.citation_accuracy)}  "
            f"Faith={fmt(scores.faithfulness)}"
            + ("  [refused]" if scores.refused else "")
            + (f"  [judge errors: {errs}]" if errs else "")
        )

    if not all_scores:
        print("\nNo cases completed. Nothing to report.")
        return

    report = aggregate(all_scores)

    log = [
        f"| Query | Precision@{args.k} | Citation Acc. | Faithfulness |",
        "|---|---|---|---|",
    ]
    for s in all_scores:
        q = (s.query[:58] + "…") if len(s.query) > 58 else s.query
        note = " — refused correctly" if s.refused and s.faithfulness is None else ""
        log.append(
            f"| {q} | {fmt(s.precision_at_k)} | {fmt(s.citation_accuracy)} | "
            f"{fmt(s.faithfulness)}{note} |"
        )

    fails = ["| Failure type | Count |", "|---|---|"]
    for ftype, count in sorted(report.failures_by_type.items(), key=lambda x: -x[1]):
        fails.append(f"| {ftype} | {count} |")

    examples = []
    for s in all_scores:
        for f in s.failures[:2]:
            examples.append(
                f"- **{f.failure_type.value}** ({f.case_id}, {f.disease or 'n/a'}): "
                f"{f.detail}" + (f"\n  - claim: “{f.claim[:120]}”" if f.claim else "")
            )

    caveat = ""
    if judge_errors:
        caveat = (
            f"\n\n> ⚠️ {judge_errors} claim(s) could not be judged (API or parsing error). "
            "They are counted as unsupported, so Faithfulness here is a LOWER BOUND."
        )

    md = "\n\n".join(
        [
            "# Evaluation Report",
            "## Scorecard",
            report.scorecard(args.k) + caveat,
            "## Evaluation Log",
            "\n".join(log),
            "## Failure Analysis",
            "\n".join(fails),
            "### Failure examples",
            "\n".join(examples[:25]) or "_No failures recorded._",
            "---",
            "_Faithfulness target 0.94 (brief) / 0.90 floor (Day 4 slide). "
            "Refused cases are excluded from Faithfulness and Citation Accuracy — "
            "a correct refusal is not a scored generation._",
        ]
    )

    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "per_case.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + report.scorecard(args.k))
    if judge_errors:
        print(f"\nWARNING: {judge_errors} claim(s) failed to be judged — see report.md")
    print(f"\nWrote {out_dir/'report.md'} and {out_dir/'per_case.json'}")


if __name__ == "__main__":
    main()
