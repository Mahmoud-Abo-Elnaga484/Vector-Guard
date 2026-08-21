
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from config import JUDGE_MODEL
from ingestion.records import ChunkRecord


# How many claims to send in a single judge call. One clinical case normally
# produces well under this, so a case costs exactly ONE judge call.
MAX_CLAIMS_PER_CALL = 15

# How much of each passage to show the judge.
PASSAGE_MAX_CHARS = 1800


JUDGE_SYSTEM = """\
You are a strict entailment judge for a medical evidence system.

You are given a set of source PASSAGES, each labelled with a chunk_id, and a numbered
list of CLAIMS. For EACH claim independently, list the chunk_ids of the passages that
support that claim.

"Supports" means: a careful reader of that passage ALONE would agree the claim is
stated in it, or follows directly and necessarily from it.

"Supports" does NOT mean:
- the passage is on the same topic as the claim
- the claim is true in general medicine
- the passage partially overlaps with the claim

If a claim contains multiple assertions, a passage supports it ONLY if it supports all
of them. If a claim is phrased as a question, or only expresses the absence of
information, set "not_a_claim": true and leave supporting_chunk_ids empty.

Judge every claim independently. Do not let your verdict on one claim influence another.

Return RAW JSON only — no markdown fences, no commentary:
{
  "judgements": [
    {
      "claim_id": "c1",
      "not_a_claim": false,
      "supporting_chunk_ids": ["..."],
      "reason": "one short sentence"
    }
  ]
}

Rules:
- Include EVERY claim_id you were given, exactly once.
- supporting_chunk_ids may only contain chunk_ids that appear verbatim in PASSAGES.
  Never invent, guess or modify a chunk_id.
- An empty supporting_chunk_ids list is a valid and often correct answer.
"""

JUDGE_USER = """\
PASSAGES:
{passages}

CLAIMS:
{claims}

Return the JSON object described in your instructions. Raw JSON only.
"""


@dataclass
class ClaimJudgement:
    claim: str
    supported_by_retrieved: bool = False
    supporting_chunk_ids: list[str] = field(default_factory=list)
    citation_verdicts: dict[str, bool] = field(default_factory=dict)
    not_a_claim: bool = False
    reason: str = ""
    error: str | None = None


def _render_passages(records: Sequence[ChunkRecord]) -> str:
    return "\n\n".join(
        f"[{r.chunk_id}]\n{r.text[:PASSAGE_MAX_CHARS]}" for r in records
    )


def _render_claims(pairs: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"{cid}: {text.strip()}" for cid, text in pairs)


def _call_judge(
    pairs: Sequence[tuple[str, str]],
    passages: str,
    attempts: int = 2,
) -> list[dict[str, Any]]:
    """One LLM call covering every claim in `pairs`. Retries once on bad JSON."""
    from generation.generator import call_llm, extract_json

    user = JUDGE_USER.format(passages=passages, claims=_render_claims(pairs))

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            raw = call_llm(JUDGE_SYSTEM, user, JUDGE_MODEL, temperature=0.0)
            data = extract_json(raw)
            return list(data.get("judgements", []) or [])
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"judge call failed: {last_error}")


def judge_claims(
    claims: Sequence[str],
    retrieved: Sequence[ChunkRecord],
    cited_map: dict[str, Sequence[str]] | None = None,
) -> dict[str, ClaimJudgement]:
    """
    Judge every claim against the retrieved passages, batching them into as few
    LLM calls as possible (normally exactly one per case).

    Returns a dict keyed by claim text, matching what metrics.score_case expects.
    """
    cited_map = cited_map or {}
    out: dict[str, ClaimJudgement] = {}

    unique_claims = list(dict.fromkeys(c for c in claims if c and c.strip()))
    if not unique_claims:
        return out

    if not retrieved:
        return {
            c: ClaimJudgement(claim=c, reason="no retrieved passages")
            for c in unique_claims
        }

    valid_ids = {r.chunk_id for r in retrieved}
    passages = _render_passages(retrieved)

    for start in range(0, len(unique_claims), MAX_CLAIMS_PER_CALL):
        batch = unique_claims[start : start + MAX_CLAIMS_PER_CALL]
        pairs = [(f"c{start + i + 1}", claim) for i, claim in enumerate(batch)]
        id_to_claim = dict(pairs)

        try:
            raw_judgements = _call_judge(pairs, passages)
        except Exception as exc:
            for claim in batch:
                out[claim] = ClaimJudgement(
                    claim=claim, error=str(exc), reason="judge call failed"
                )
            continue

        for j in raw_judgements:
            if not isinstance(j, dict):
                continue
            claim = id_to_claim.get(str(j.get("claim_id", "")).strip())
            if claim is None or claim in out:
                continue

            # Drop any chunk_id the judge invented — same defensive stance as
            # the generator's citation hydration.
            supporting = [
                str(s)
                for s in (j.get("supporting_chunk_ids") or [])
                if str(s) in valid_ids
            ]
            cited = list(dict.fromkeys(cited_map.get(claim, [])))

            out[claim] = ClaimJudgement(
                claim=claim,
                supported_by_retrieved=bool(supporting),
                supporting_chunk_ids=supporting,
                citation_verdicts={c: (c in supporting) for c in cited},
                not_a_claim=bool(j.get("not_a_claim", False)),
                reason=str(j.get("reason", ""))[:200],
            )

        # The judge silently skipped a claim — record it rather than losing it.
        for claim in batch:
            if claim not in out:
                out[claim] = ClaimJudgement(
                    claim=claim,
                    error="missing from judge output",
                    reason="the judge omitted this claim",
                )

    return out


def judge_all_claims(response: Any, retrieved: Sequence[ChunkRecord]) -> dict[str, ClaimJudgement]:
    """Judge every claim in a DifferentialDiagnosisResponse in one batched pass."""
    cited_map: dict[str, list[str]] = {}
    order: list[str] = []

    for _disease, _field_name, item in response.all_claims():
        claim = item.claim
        if claim not in cited_map:
            cited_map[claim] = []
            order.append(claim)
        cited_map[claim] = list(
            dict.fromkeys(cited_map[claim] + [c.chunk_id for c in item.citations])
        )

    return judge_claims(order, retrieved, cited_map)


def judge_claim(
    claim: str,
    retrieved: Sequence[ChunkRecord],
    cited_chunk_ids: Sequence[str] = (),
) -> ClaimJudgement:
    """Single-claim helper, kept so older callers keep working."""
    result = judge_claims([claim], retrieved, {claim: list(cited_chunk_ids)})
    return result.get(claim, ClaimJudgement(claim=claim, error="no judgement returned"))
