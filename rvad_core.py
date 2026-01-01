import random
import re
from typing import Any, Dict, List, Tuple

NONVISUAL_PATTERNS = [
    r"\bsterile\b",
    r"\bbacteria-free\b",
    r"\borganic\b",
    r"\bpesticide-free\b",
    r"\blab-tested\b",
]

SENTINEL_SCOPE = [
    "allergen_nut",
    "allergen_seafood",
    "allergen_dairy",
    "allergen_gluten",
    "regulated_cannabis",
    "regulated_medication",
]


def _contains_any(text: str, patterns: List[str]) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in patterns)


def extract_claims(text: str) -> Tuple[List[Dict[str, Any]], bool]:
    claims: List[Dict[str, Any]] = []
    nonvisual_claim = _contains_any(text, NONVISUAL_PATTERNS)
    return claims, nonvisual_claim


def detect_nonvisual_claim(text: str) -> bool:
    return _contains_any(text, NONVISUAL_PATTERNS)


def mock_detect_sentinel(fp_rate: float, rng: random.Random, scope_items: List[str]) -> List[str]:
    detected = []
    for item in scope_items:
        if rng.random() < fp_rate:
            detected.append(item)
    return detected


def extract_claims_regex(text: str, scope_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    lower = text.lower()
    for risk_item, config in scope_config.items():
        triggers = config.get("claim_triggers", [])
        for trig in triggers:
            if trig.lower() in lower:
                claims.append(
                    {
                        "risk_item": risk_item,
                        "polarity": "ASSERT_ABSENT",
                        "explicit": True,
                    }
                )
                break
    return claims


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_sentinel_queries(scope_config: Dict[str, Any], claims: List[Dict[str, Any]]):
    queries: List[str] = []
    extra_claim_queries: Dict[str, List[str]] = {}
    for risk_item, config in scope_config.items():
        queries.extend(config.get("queries", []))

    claim_items = {
        c.get("risk_item")
        for c in claims
        if c.get("polarity") == "ASSERT_ABSENT"
    }
    for risk_item in claim_items:
        config = scope_config.get(risk_item, {})
        extra = config.get("claim_queries", [])
        if extra:
            extra_claim_queries[risk_item] = list(extra)
            queries.extend(extra)

    return _dedupe_preserve_order(queries), extra_claim_queries


def map_sentinel_results(
    scope_config: Dict[str, Any],
    results: Dict[str, Any],
    extra_claim_queries=None,
):
    detected = []
    uncertain_items = []
    evidence = []
    extra_claim_queries = extra_claim_queries or {}
    for risk_item, config in scope_config.items():
        local_queries = list(config.get("queries", []))
        local_queries.extend(extra_claim_queries.get(risk_item, []))
        min_conf = float(config.get("min_confidence", 0.5))
        present = False
        uncertain = False
        best = None
        for q in local_queries:
            res = results.get(q, {})
            confidence = res.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            if res.get("present") is True and confidence >= min_conf:
                present = True
                if not best or confidence >= float(best.get("confidence", 0.0) or 0.0):
                    best = res
            elif res.get("present") == "uncertain" and confidence >= 0.4:
                uncertain = True
                if not best or confidence >= float(best.get("confidence", 0.0) or 0.0):
                    best = res
            elif res.get("present") is True and confidence < min_conf:
                uncertain = True
                if not best or confidence >= float(best.get("confidence", 0.0) or 0.0):
                    best = res
        # Aggregation rule: risk_item is present iff any query present=True
        # with confidence >= min_confidence. Uncertain blocks CONSISTENT.
        if present:
            detected.append(risk_item)
            evidence.append(
                {
                    "risk_item": risk_item,
                    "present": True,
                    "bbox": None,
                    "score": float(best.get("confidence", 0.0)) if best else 0.0,
                    "source": "mllm",
                    "evidence": best.get("evidence", "") if best else "",
                }
            )
        elif uncertain:
            uncertain_items.append(risk_item)
            evidence.append(
                {
                    "risk_item": risk_item,
                    "present": "uncertain",
                    "bbox": None,
                    "score": float(best.get("confidence", 0.0)) if best else 0.0,
                    "source": "mllm",
                    "evidence": best.get("evidence", "") if best else "",
                }
            )
        else:
            evidence.append(
                {
                    "risk_item": risk_item,
                    "present": False,
                    "bbox": None,
                    "score": float(best.get("confidence", 0.0)) if best else 0.0,
                    "source": "mllm",
                    "evidence": best.get("evidence", "") if best else "",
                }
            )

    return detected, uncertain_items, evidence


def decide_verdict(
    claims: List[Dict[str, Any]],
    nonvisual_claim: bool,
    sentinel_detected,
    sentinel_uncertain,
):
    verdict = "CONSISTENT"
    reason = "NONE"

    claim_items = {
        c.get("risk_item")
        for c in claims
        if c.get("polarity") == "ASSERT_ABSENT"
    }
    contradiction_items = [item for item in sentinel_detected if item in claim_items]
    omission_items = [item for item in sentinel_detected if item not in claim_items]

    # Priority order is fixed to match the paper narrative.
    if contradiction_items:
        verdict = "ALERT"
        reason = "CONTRADICTION"
    elif omission_items:
        verdict = "ALERT"
        reason = "CRITICAL_OMISSION"
    elif sentinel_uncertain:
        verdict = "UNVERIFIED"
        reason = "SENTINEL_UNCERTAIN"
    elif nonvisual_claim:
        verdict = "UNVERIFIED"
        reason = "NONVISUAL_UNVERIFIED"

    return verdict, reason


def dedupe_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple, Dict[str, Any]] = {}
    for claim in claims:
        key = (
            claim.get("risk_item"),
            claim.get("polarity"),
            bool(claim.get("explicit")),
        )
        entry = merged.get(key)
        span = claim.get("span")
        if entry is None:
            entry = {
                "risk_item": key[0],
                "polarity": key[1],
                "explicit": key[2],
            }
            merged[key] = entry
        if span:
            spans = entry.get("span")
            if spans:
                merged_spans = {s.strip() for s in spans.split(" | ") if s.strip()}
            else:
                merged_spans = set()
            merged_spans.add(str(span).strip())
            entry["span"] = " | ".join(sorted(merged_spans))
    return list(merged.values())
