

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

from openai import OpenAI

from config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_JSON_MODE,
    LLM_MAX_RETRIES,
    LLM_MIN_INTERVAL,
    TOP_K,
)
from ingestion.records import ChunkRecord
from retrieval.retriever import format_evidence_block, search_documents

from generation.guardrails import (
    ClaimAudit,
    RiskLevel,
    audit_and_repair,
    classify_input,
    gate_retrieval,
)
from generation.prompts import GROUNDING_SYSTEM_PROMPT, build_user_message
from generation.schemas import (
    Citation,
    DifferentialDiagnosisResponse,
    DiseaseAssessment,
    EvidenceItem,
    LLMDifferentialResponse,
    empty_assessment,
    refusal_response,
)

_client: OpenAI | None = None
_last_call_at: float = 0.0

_json_mode_supported: bool = LLM_JSON_MODE


def get_client() -> OpenAI:

    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY is missing from .env "
                "(or OPENAI_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY)"
            )
        kwargs: dict[str, Any] = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _client = OpenAI(**kwargs)
    return _client


def _throttle() -> None:
    
    global _last_call_at
    if LLM_MIN_INTERVAL <= 0:
        return
    wait = LLM_MIN_INTERVAL - (time.time() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.time()


def call_llm(system: str, user: str, model: str, temperature: float = 0.0) -> str:
    
    global _json_mode_supported

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        _throttle()
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if _json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content or "{}"
        except Exception as exc:
            last_error = exc
            text = str(exc).lower()

           
            if _json_mode_supported and "response_format" in text:
                _json_mode_supported = False
                print("[LLM] Provider rejected JSON mode — falling back to prompt-only JSON.")
                continue

           
            if "rate" in text or "429" in text or "quota" in text:
                delay = 2 ** attempt * 5
                print(f"[LLM] Rate limited. Waiting {delay}s (attempt {attempt+1}/{LLM_MAX_RETRIES})...")
                time.sleep(delay)
                continue

            raise

    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last_error}")


def extract_json(raw: str) -> dict[str, Any]:

    text = raw.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = max(blocks, key=len).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No JSON object found in model output: {raw[:200]}")


def _parse_llm_json(raw: str) -> LLMDifferentialResponse:
    return LLMDifferentialResponse.model_validate(extract_json(raw))





def _hydrate_items(items: Sequence[Any], index: dict[str, ChunkRecord]) -> list[EvidenceItem]:

    out: list[EvidenceItem] = []
    for it in items:
        citations, rejected = [], []
        for cid in dict.fromkeys(it.chunk_ids):  
            rec = index.get(cid)
            if rec is None:
                rejected.append(cid)
                continue
            citations.append(
                Citation(
                    chunk_id=rec.chunk_id,
                    document_name=rec.document_name,
                    section_title=rec.section_title,
                    page_number=rec.page_number,
                    source_url=rec.source_url,
                )
            )
        out.append(
            EvidenceItem(claim=it.claim.strip(), citations=citations, rejected_chunk_ids=rejected)
        )
    return out


def _hydrate_assessment(llm_a: Any, index: dict[str, ChunkRecord]) -> DiseaseAssessment:
    return DiseaseAssessment(
        disease=llm_a.disease,
        evidence_for=_hydrate_items(llm_a.evidence_for, index),
        evidence_against=_hydrate_items(llm_a.evidence_against, index),
        missing_information=[m.strip() for m in llm_a.missing_information if m.strip()],
        next_questions=[q.strip() for q in llm_a.next_questions if q.strip()],
        tests=_hydrate_items(llm_a.tests, index),
        support_level=llm_a.support_level,
    )

def generate_differential(
    query: str,
    top_k: int = TOP_K,
    memory_block: str = "",
    apply_input_guardrail: bool = True,
    return_debug: bool = False,
) -> DifferentialDiagnosisResponse | tuple[DifferentialDiagnosisResponse, dict[str, Any]]:

    debug: dict[str, Any] = {"query": query}

    caution = ""
    if apply_input_guardrail:
        decision = classify_input(query)
        debug["input_risk"] = decision.level.value
        debug["input_reason"] = decision.reason
        if decision.level is RiskLevel.REFUSE:
            resp = refusal_response(query, decision.message)
            return (resp, debug) if return_debug else resp
        if decision.level is RiskLevel.NEEDS_CAUTION:
            caution = decision.message

    records = search_documents(query, top_k=top_k)
    index = {r.chunk_id: r for r in records}
    debug["retrieved"] = [
        {"chunk_id": r.chunk_id, "section": r.section_title, "score": r.score} for r in records
    ]

    gate = gate_retrieval(records)
    debug["retrieval_gate"] = gate.reason
    debug["top_score"] = gate.top_score
    if not gate.allow_generation:
        resp = refusal_response(
            query,
            "Insufficient evidence from the retrieved sources. No indexed passage met the "
            f"minimum retrieval confidence threshold (top score: {gate.top_score}).",
        )
        resp.retrieved_chunk_ids = [r.chunk_id for r in records]
        resp.top_score = gate.top_score
        return (resp, debug) if return_debug else resp

    user_msg = build_user_message(
        query=query,
        evidence_block=format_evidence_block(records),
        memory_block=memory_block,
    )
    raw = call_llm(
        GROUNDING_SYSTEM_PROMPT, user_msg, GENERATION_MODEL, GENERATION_TEMPERATURE
    )
    debug["raw_llm_output"] = raw

    try:
        llm_resp = _parse_llm_json(raw)
    except Exception as exc:  
        debug["parse_error"] = str(exc)
        raw = call_llm(
            GROUNDING_SYSTEM_PROMPT,
            user_msg + f"\n\nYour previous output was invalid JSON ({exc}). Return valid raw JSON only.",
            GENERATION_MODEL,
            0.0,
        )
        llm_resp = _parse_llm_json(raw)

    response = DifferentialDiagnosisResponse(
        query=query,
        dengue=_hydrate_assessment(llm_resp.dengue, index),
        chikungunya=_hydrate_assessment(llm_resp.chikungunya, index),
        zika=_hydrate_assessment(llm_resp.zika, index),
        yellow_fever=_hydrate_assessment(llm_resp.yellow_fever, index),
        overall_summary=llm_resp.overall_summary.strip(),
        confidence=llm_resp.confidence,
        retrieved_chunk_ids=[r.chunk_id for r in records],
        top_score=gate.top_score,
    )

    audit: ClaimAudit = audit_and_repair(response, records)
    debug["claim_audit"] = {
        "total_claims": audit.total_claims,
        "uncited_claims": audit.uncited_claims,
        "hallucinated_chunk_ids": audit.hallucinated_chunk_ids,
        "dropped_claims": audit.dropped_claims,
        "details": audit.details,
    }

    if gate.confidence_cap == "Low" and response.confidence in ("High", "Medium"):
        response.confidence = "Low"
        debug["confidence_downgraded"] = True

    if caution:
        response.overall_summary = f"{caution}\n\n{response.overall_summary}".strip()

    return (response, debug) if return_debug else response


def generate_response(
    messages: list[dict[str, str]] | str, context: Any = None
) -> tuple[str, Any, Any]:
    """
    High-level entry point for Streamlit UI integrations. Accepts message history or a query string,
    executes the evidence-grounded differential diagnosis pipeline, and returns formatted markdown,
    the raw response object, and the retrieved chunks for live evaluation.
    """
    if isinstance(messages, str):
        query = messages
        memory_block = ""
    elif isinstance(messages, list) and len(messages) > 0:
        query = messages[-1].get("content", "")
        earlier_facts = [
            m["content"] for m in messages[:-1] if isinstance(m, dict) and m.get("role") == "user"
        ]
        if earlier_facts:
            lines = "\n".join(f"- {fact}" for fact in earlier_facts)
            memory_block = (
                "### SESSION MEMORY\n"
                "Reported by the user in earlier turns. This is case state, NOT retrieved "
                "evidence. Never cite it and never treat it as guideline support.\n"
                f"{lines}"
            )
        else:
            memory_block = ""
    else:
        query = ""
        memory_block = ""

    if not query:
        # لازم نرجع 3 حاجات هنا برضه عشان الواجهة متضربش إيرور
        return "Please enter a clinical presentation or question.", None, []

    # 1. توليد الإجابة
    resp = generate_differential(query, memory_block=memory_block)
    if isinstance(resp, tuple):
        resp = resp[0]
        
    # 2. جلب النصوص اللي تم استخدامها في الإجابة عشان نبعتها للتقييم
    retrieved_chunks = search_documents(query, top_k=TOP_K)
    
    # 3. إرجاع الـ 3 متغيرات اللي الواجهة مستنياهم
    return resp.to_markdown(), resp, retrieved_chunks


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or (
        "28-year-old with abrupt fever for 4 days, severe symmetrical joint pain in "
        "hands and ankles, and a maculopapular rash. No bleeding."
    )
    resp, dbg = generate_differential(q, return_debug=True)  # type: ignore[misc]
    print(resp.to_markdown())
    print("\n--- DEBUG ---")
    print(json.dumps({k: v for k, v in dbg.items() if k != "raw_llm_output"}, indent=2))
