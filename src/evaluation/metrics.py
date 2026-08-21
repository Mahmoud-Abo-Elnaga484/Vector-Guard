

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from ingestion.records import ChunkRecord
from generation.schemas import DifferentialDiagnosisResponse

FAITHFULNESS_TARGET = 0.94
FAITHFULNESS_FLOOR = 0.90
CITATION_ACCURACY_TARGET = 0.90
PRECISION_TARGET = 0.60





def precision_at_k(
    retrieved: Sequence[ChunkRecord],
    k: int,
    relevant_chunk_ids: Sequence[str] = (),
    relevant_sections: Sequence[str] = (),
) -> float | None:

    if not relevant_chunk_ids and not relevant_sections:
        return None

    ids = set(relevant_chunk_ids)
    sections = {s.strip().lower() for s in relevant_sections}

    def is_relevant(r) -> bool:
        if r.chunk_id in ids:
            return True
        # Match anywhere in the header path, not just the deepest header.
        # A chunk under "4.2 Symptom control > 4.2.1.1 Evidence to decision" has
        # section_title "4.2.1.1 Evidence to decision" but is genuinely part of
        # section 4.2, so annotating at the section level has to count it.
        candidates = [r.section_title or ""] + list(getattr(r, "header_path", []) or [])
        return any(c.strip().lower() in sections for c in candidates if c)

    top = list(retrieved)[:k]
    hits = sum(1 for r in top if is_relevant(r))
    return hits / k if k else 0.0





@dataclass
class CaseScores:
    case_id: str
    query: str
    refused: bool = False

    precision_at_k: float | None = None
    citation_accuracy: float | None = None
    faithfulness: float | None = None

    total_claims: int = 0
    supported_claims: int = 0
    total_citations: int = 0
    correct_citations: int = 0

    failures: list["FailureRecord"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def unsupported_claim_rate(self) -> float | None:
        return None if self.faithfulness is None else 1.0 - self.faithfulness


class FailureType(str, Enum):
    RETRIEVAL = "retrieval_failure"          
    GENERATION = "generation_failure"        
    CITATION = "citation_failure"            
    METADATA = "metadata_failure"            
    INSUFFICIENT_OK = "insufficient_evidence_correct"   
    CONFLICTING = "conflicting_evidence"     


@dataclass
class FailureRecord:
    case_id: str
    failure_type: FailureType
    disease: str = ""
    claim: str = ""
    detail: str = ""


def score_case(
    case_id: str,
    response: DifferentialDiagnosisResponse,
    retrieved: Sequence[ChunkRecord],
    judgements: dict[str, "object"],
    k: int,
    relevant_chunk_ids: Sequence[str] = (),
    relevant_sections: Sequence[str] = (),
    expected_behavior: str = "answer",
) -> CaseScores:

    scores = CaseScores(case_id=case_id, query=response.query, refused=response.refused)
    scores.precision_at_k = precision_at_k(
        retrieved, k, relevant_chunk_ids, relevant_sections
    )

    
    if response.refused:
        if expected_behavior == "refuse":
            scores.notes.append("Correct refusal — counted as N/A, not as a failure.")
            scores.failures.append(
                FailureRecord(case_id, FailureType.INSUFFICIENT_OK, detail=response.refusal_reason or "")
            )
        else:
            scores.notes.append("Refused a case that should have been answered.")
            scores.failures.append(
                FailureRecord(
                    case_id,
                    FailureType.RETRIEVAL,
                    detail=f"Unexpected refusal: {response.refusal_reason}",
                )
            )
        return scores  

    index = {r.chunk_id: r for r in retrieved}

    for disease, field_name, item in response.all_claims():
        j = judgements.get(item.claim)
        if j is None or getattr(j, "not_a_claim", False):
            continue

        scores.total_claims += 1
        supported = bool(getattr(j, "supported_by_retrieved", False))
        if supported:
            scores.supported_claims += 1
        else:
            scores.failures.append(
                FailureRecord(
                    case_id,
                    FailureType.GENERATION,
                    disease,
                    item.claim,
                    getattr(j, "reason", "") or "no retrieved passage supports this claim",
                )
            )

        verdicts = getattr(j, "citation_verdicts", {}) or {}

        for c in item.citations:
            scores.total_citations += 1
            rec = index.get(c.chunk_id)

            
            if rec is None:
                scores.failures.append(
                    FailureRecord(case_id, FailureType.CITATION, disease, item.claim,
                                  f"chunk_id {c.chunk_id} not in retrieved set")
                )
                continue
            metadata_ok = (
                c.document_name == rec.document_name
                and c.section_title == rec.section_title
                and c.page_number == rec.page_number
            )
            if not metadata_ok:
                scores.failures.append(
                    FailureRecord(case_id, FailureType.METADATA, disease, item.claim,
                                  f"metadata mismatch for {c.chunk_id}")
                )
                continue

            
            if verdicts.get(c.chunk_id, False):
                scores.correct_citations += 1
            else:
                ftype = (
                    FailureType.CITATION if supported else FailureType.GENERATION
                )
                scores.failures.append(
                    FailureRecord(
                        case_id, ftype, disease, item.claim,
                        f"cited {c.chunk_id} does not support the claim"
                        + (
                            f"; actually supported by {getattr(j, 'supporting_chunk_ids', [])}"
                            if supported else ""
                        ),
                    )
                )

    scores.faithfulness = (
        scores.supported_claims / scores.total_claims if scores.total_claims else None
    )
    scores.citation_accuracy = (
        scores.correct_citations / scores.total_citations if scores.total_citations else None
    )
    return scores




def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass
class DatasetReport:
    n_cases: int
    n_scored: int
    n_refused: int
    precision_at_k: float | None
    citation_accuracy: float | None
    faithfulness: float | None
    unsupported_claim_rate: float | None
    failures_by_type: dict[str, int]

    def scorecard(self, k: int) -> str:
        def row(name, actual, target):
            if actual is None:
                return f"| {name} | N/A | {target} | — |"
            status = "PASS" if actual >= target else "FAIL"
            return f"| {name} | {actual:.3f} | {target:.2f} | {status} |"

        lines = [
            "| Metric | Actual | Target | Pass/Fail |",
            "|---|---|---|---|",
            row(f"Retrieval Precision@{k}", self.precision_at_k, PRECISION_TARGET),
            row("Citation Accuracy", self.citation_accuracy, CITATION_ACCURACY_TARGET),
            row("Faithfulness", self.faithfulness, FAITHFULNESS_TARGET),
        ]
        if self.unsupported_claim_rate is not None:
            lines.append(
                f"| Unsupported Claim Rate (optional) | {self.unsupported_claim_rate:.3f} | ≤0.06 | "
                f"{'PASS' if self.unsupported_claim_rate <= 0.06 else 'FAIL'} |"
            )
        lines.append("")
        lines.append(
            f"Cases: {self.n_cases} total · {self.n_scored} scored · {self.n_refused} refused"
        )
        return "\n".join(lines)


def aggregate(cases: Sequence[CaseScores]) -> DatasetReport:
    by_type: dict[str, int] = {}
    for c in cases:
        for f in c.failures:
            by_type[f.failure_type.value] = by_type.get(f.failure_type.value, 0) + 1

    faith = _mean([c.faithfulness for c in cases])
    return DatasetReport(
        n_cases=len(cases),
        n_scored=sum(1 for c in cases if not c.refused),
        n_refused=sum(1 for c in cases if c.refused),
        precision_at_k=_mean([c.precision_at_k for c in cases]),
        citation_accuracy=_mean([c.citation_accuracy for c in cases]),
        faithfulness=faith,
        unsupported_claim_rate=None if faith is None else 1.0 - faith,
        failures_by_type=by_type,
    )
