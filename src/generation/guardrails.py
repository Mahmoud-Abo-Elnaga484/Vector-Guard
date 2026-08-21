
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ingestion.records import ChunkRecord
from generation.schemas import DifferentialDiagnosisResponse, EvidenceItem

class RiskLevel(str, Enum):
    ALLOWED = "allowed"
    NEEDS_CAUTION = "needs_caution"
    REFUSE = "refuse"


@dataclass
class RiskDecision:
    level: RiskLevel
    reason: str
    message: str = ""

_EMERGENCY = re.compile(
    r"\b(unconscious|unresponsive|not breathing|cardiac arrest|seizure now|"
    r"massive bleeding|haemorrhagic shock|hemorrhagic shock|severe shock|"
    r"suicide|kill myself|overdosed?)\b",
    re.I,
)

_PRESCRIBING = re.compile(
    r"\b(how many mg|what dose should i (take|give)|prescribe me|"
    r"can i take \d+|dosage for my)\b",
    re.I,
)

_IN_SCOPE = re.compile(
    r"\b(dengue|chikungunya|zika|arbovir\w*|yellow fever|mosquito|aedes|"
    r"fever|rash|arthralgia|joint pain|myalgia|thrombocytopeni\w*|"
    r"platelet|haematocrit|hematocrit|warning signs?|plasma leakage|"
    r"tourniquet|NS1|RT-PCR|serolog\w*|IgM|IgG)\b",
    re.I,
)

_PATIENT_SCENARIO = re.compile(
    r"\b(my (patient|son|daughter|wife|husband|mother|father|child)|"
    r"i have|i feel|i am experiencing|should i go to)\b",
    re.I,
)

REFUSAL_EMERGENCY = (
    "This description suggests a potential medical emergency. This tool does not "
    "handle emergencies and cannot triage acute deterioration. Please contact "
    "emergency services or present to the nearest emergency department immediately."
)

REFUSAL_OUT_OF_SCOPE = (
    "This query falls outside the scope of the indexed source library, which covers "
    "WHO guidance on arboviral diseases (dengue, chikungunya, Zika, yellow fever). "
    "Insufficient evidence from the retrieved sources."
)

REFUSAL_PRESCRIBING = (
    "This tool retrieves and organizes guideline evidence; it does not provide dosing "
    "or individual treatment instructions. Please consult a qualified clinician."
)

CAUTION_NOTE = (
    "Note: this appears to describe a specific individual. The output below organizes "
    "guideline evidence only and is not an assessment of that person."
)


def classify_input(query: str) -> RiskDecision:
    q = (query or "").strip()

    if not q:
        return RiskDecision(RiskLevel.REFUSE, "empty_query", "No query was provided.")

    if _EMERGENCY.search(q):
        return RiskDecision(RiskLevel.REFUSE, "emergency", REFUSAL_EMERGENCY)

    if _PRESCRIBING.search(q):
        return RiskDecision(RiskLevel.REFUSE, "prescribing_request", REFUSAL_PRESCRIBING)

    if not _IN_SCOPE.search(q):
        return RiskDecision(RiskLevel.REFUSE, "out_of_scope", REFUSAL_OUT_OF_SCOPE)

    if _PATIENT_SCENARIO.search(q):
        return RiskDecision(RiskLevel.NEEDS_CAUTION, "patient_scenario", CAUTION_NOTE)

    return RiskDecision(RiskLevel.ALLOWED, "in_scope")

BLOCK_THRESHOLD = 0.30   
LOW_THRESHOLD = 0.45     

@dataclass
class RetrievalGate:
    allow_generation: bool
    reason: str
    confidence_cap: str | None = None
    top_score: float | None = None


def gate_retrieval(records: Sequence[ChunkRecord]) -> RetrievalGate:
    if not records:
        return RetrievalGate(False, "no_chunks_retrieved", top_score=None)

    scores = [r.score for r in records if r.score is not None]
    top = max(scores) if scores else None

    if top is None:
        return RetrievalGate(True, "scores_unavailable")

    if top < BLOCK_THRESHOLD:
        return RetrievalGate(False, "below_block_threshold", top_score=top)

    if top < LOW_THRESHOLD:
        return RetrievalGate(True, "weak_retrieval", confidence_cap="Low", top_score=top)

    return RetrievalGate(True, "ok", top_score=top)

@dataclass
class ClaimAudit:
    total_claims: int = 0
    uncited_claims: int = 0
    hallucinated_chunk_ids: int = 0
    dropped_claims: int = 0
    misattributed_claims: int = 0
    duplicated_lists: int = 0
    details: list[str] = None  

    def __post_init__(self):
        if self.details is None:
            self.details = []

_DISEASE_TERMS = {
    "Dengue": ("dengue",),
    "Chikungunya": ("chikungunya", "chik"),
    "Zika": ("zika",),
    "Yellow Fever": ("yellow fever", "yellow-fever"),
}


def _misattributed(claim: str, disease: str) -> str | None:

    text = claim.lower()
    own = any(t in text for t in _DISEASE_TERMS.get(disease, ()))
    if own:
        return None
    for other, terms in _DISEASE_TERMS.items():
        if other != disease and any(t in text for t in terms):
            return other
    return None


def _lists_are_duplicated(response: DifferentialDiagnosisResponse, field_name: str) -> bool:
    
    signatures = []
    for a in response.assessments():
        value = getattr(a, field_name)
        items = [i.claim if hasattr(i, "claim") else str(i) for i in value]
        if items:
            signatures.append(tuple(sorted(s.strip().lower() for s in items)))
    return len(signatures) > 1 and len(set(signatures)) < len(signatures)


def audit_and_repair(
    response: DifferentialDiagnosisResponse,
    retrieved: Sequence[ChunkRecord],
    drop_uncited: bool = True,
) -> ClaimAudit:
    valid_ids = {r.chunk_id for r in retrieved}
    audit = ClaimAudit()

    for assessment in response.assessments():
        for field_name in ("evidence_for", "evidence_against", "tests"):
            kept: list[EvidenceItem] = []
            for item in getattr(assessment, field_name):
                audit.total_claims += 1

                bad = [c for c in item.citations if c.chunk_id not in valid_ids]
                if bad:
                    audit.hallucinated_chunk_ids += len(bad)
                    audit.details.append(
                        f"{assessment.disease}/{field_name}: dropped invalid chunk_ids "
                        f"{[c.chunk_id for c in bad]}"
                    )
                    item.rejected_chunk_ids += [c.chunk_id for c in bad]
                    item.citations = [c for c in item.citations if c.chunk_id in valid_ids]

                if not item.citations:
                    audit.uncited_claims += 1
                    if drop_uncited:
                        audit.dropped_claims += 1
                        audit.details.append(
                            f"{assessment.disease}/{field_name}: dropped uncited claim "
                            f"“{item.claim[:70]}…”"
                        )
                        continue
                wrong = _misattributed(item.claim, assessment.disease)
                if wrong:
                    audit.misattributed_claims += 1
                    audit.dropped_claims += 1
                    audit.details.append(
                        f"{assessment.disease}/{field_name}: dropped claim about "
                        f"{wrong} placed under {assessment.disease} — "
                        f"“{item.claim[:70]}…”"
                    )
                    continue

                kept.append(item)

            setattr(assessment, field_name, kept)

        if not assessment.evidence_for and not assessment.evidence_against:
            assessment.support_level = "insufficient_evidence"

    for field_name in ("tests", "missing_information", "next_questions"):
        if _lists_are_duplicated(response, field_name):
            audit.duplicated_lists += 1
            audit.details.append(
                f"WARNING: '{field_name}' is identical across two or more diseases "
                f"— the differential is not actually differentiating."
            )

    if all(a.support_level == "insufficient_evidence" for a in response.assessments()):
        response.confidence = "Insufficient Evidence"

    distinct = {c.chunk_id for _d, _i, c in response.all_citations()}
    if len(distinct) <= 2 and response.confidence in ("High", "Medium"):
        response.confidence = "Low"
        audit.details.append(
            f"Confidence capped to Low: only {len(distinct)} distinct chunk(s) cited."
        )
    elif len(distinct) <= 4 and response.confidence == "High":
        response.confidence = "Medium"
        audit.details.append(
            f"Confidence capped to Medium: only {len(distinct)} distinct chunks cited."
        )

    return audit
