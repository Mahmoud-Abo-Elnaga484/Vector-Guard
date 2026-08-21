
from __future__ import annotations

from typing import Iterator, Literal, Optional

from pydantic import BaseModel, Field

DiseaseName = Literal["Dengue", "Chikungunya", "Zika", "Yellow Fever"]
SupportLevel = Literal[
    "supported", "partially_supported", "not_supported", "insufficient_evidence"
]
ConfidenceLevel = Literal["High", "Medium", "Low", "Insufficient Evidence"]

INSUFFICIENT = "Insufficient evidence from the retrieved sources."
NO_COUNTER_EVIDENCE = (
    "No retrieved evidence was found that directly argues against this diagnosis."
)
DISCLAIMER = (
    "This system supports — never replaces — clinical judgment. Outputs are "
    "guideline-grounded, not diagnostic."
)
class LLMEvidenceItem(BaseModel):
    claim: str = Field(description="A single atomic statement grounded in retrieved text.")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of retrieved chunks that support this exact claim.",
    )


class LLMDiseaseAssessment(BaseModel):
    disease: DiseaseName
    evidence_for: list[LLMEvidenceItem] = Field(default_factory=list)
    evidence_against: list[LLMEvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)
    tests: list[LLMEvidenceItem] = Field(default_factory=list)
    support_level: SupportLevel = "insufficient_evidence"


class LLMDifferentialResponse(BaseModel):
    dengue: LLMDiseaseAssessment
    chikungunya: LLMDiseaseAssessment
    zika: LLMDiseaseAssessment
    yellow_fever: LLMDiseaseAssessment
    overall_summary: str = ""
    confidence: ConfidenceLevel = "Insufficient Evidence"

class Citation(BaseModel):
    chunk_id: str
    document_name: str
    section_title: str
    page_number: Optional[int] = None
    source_url: Optional[str] = None

    def label(self) -> str:
        page = f"p. {self.page_number}" if self.page_number is not None else "page n/a"
        return f"{self.document_name} — § {self.section_title} — {page} [{self.chunk_id}]"


class EvidenceItem(BaseModel):
    claim: str
    citations: list[Citation] = Field(default_factory=list)
    
    rejected_chunk_ids: list[str] = Field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return len(self.citations) > 0


class DiseaseAssessment(BaseModel):
    disease: DiseaseName
    evidence_for: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)
    tests: list[EvidenceItem] = Field(default_factory=list)
    support_level: SupportLevel = "insufficient_evidence"

    def claim_items(self) -> Iterator[tuple[str, EvidenceItem]]:

        for field_name in ("evidence_for", "evidence_against", "tests"):
            for item in getattr(self, field_name):
                yield field_name, item


class DifferentialDiagnosisResponse(BaseModel):
    query: str
    dengue: DiseaseAssessment
    chikungunya: DiseaseAssessment
    zika: DiseaseAssessment
    yellow_fever: DiseaseAssessment
    overall_summary: str = ""
    confidence: ConfidenceLevel = "Insufficient Evidence"
    disclaimer: str = DISCLAIMER

    refused: bool = False
    refusal_reason: Optional[str] = None

    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    top_score: Optional[float] = None

    def assessments(self) -> list[DiseaseAssessment]:
        return [self.dengue, self.chikungunya, self.zika, self.yellow_fever]

    def all_claims(self) -> Iterator[tuple[str, str, EvidenceItem]]:
        
        for a in self.assessments():
            for field_name, item in a.claim_items():
                yield a.disease, field_name, item

    def all_citations(self) -> Iterator[tuple[str, EvidenceItem, Citation]]:
        for disease, _field, item in self.all_claims():
            for c in item.citations:
                yield disease, item, c

    def to_markdown(self) -> str:
        
        if self.refused:
            return f"### Request not answered\n\n{self.refusal_reason}\n\n_{self.disclaimer}_"

        out = [f"## Differential Diagnosis\n\n**Query:** {self.query}\n"]
        if self.overall_summary:
            out.append(f"{self.overall_summary}\n")
        out.append(f"**Confidence:** {self.confidence}\n")

        icons = {"evidence_for": "✓", "evidence_against": "✗", "tests": "🧪"}
        titles = {
            "evidence_for": "Evidence FOR",
            "evidence_against": "Evidence AGAINST",
            "tests": "Test",
        }

        for a in self.assessments():
            out.append(f"\n### {a.disease}  _({a.support_level.replace('_', ' ')})_\n")
            for field_name in ("evidence_for", "evidence_against", "tests"):
                items = getattr(a, field_name)
                out.append(f"**{icons[field_name]} {titles[field_name]}**")
                if not items:
                    filler = (
                        NO_COUNTER_EVIDENCE if field_name == "evidence_against" else INSUFFICIENT
                    )
                    out.append(f"- {filler}")
                for it in items:
                    cites = " ".join(f"`[{c.chunk_id}]`" for c in it.citations) or "`[uncited]`"
                    out.append(f"- {it.claim} {cites}")
                out.append("")

            out.append("**? Missing Information**")
            out += [f"- {m}" for m in a.missing_information] or [f"- {INSUFFICIENT}"]
            out.append("\n**→ Next Question**")
            out += [f"- {q}" for q in a.next_questions] or [f"- {INSUFFICIENT}"]
            out.append("")

        seen: dict[str, Citation] = {}
        for _d, _i, c in self.all_citations():
            seen.setdefault(c.chunk_id, c)
        if seen:
            out.append("\n### 📚 Citations\n")
            out += [f"- {c.label()}" for c in seen.values()]

        out.append(f"\n---\n_{self.disclaimer}_")
        return "\n".join(out)


def empty_assessment(disease: DiseaseName) -> DiseaseAssessment:
    return DiseaseAssessment(disease=disease, support_level="insufficient_evidence")


def refusal_response(query: str, reason: str) -> DifferentialDiagnosisResponse:
    return DifferentialDiagnosisResponse(
        query=query,
        dengue=empty_assessment("Dengue"),
        chikungunya=empty_assessment("Chikungunya"),
        zika=empty_assessment("Zika"),
        yellow_fever=empty_assessment("Yellow Fever"),
        overall_summary="",
        confidence="Insufficient Evidence",
        refused=True,
        refusal_reason=reason,
    )
