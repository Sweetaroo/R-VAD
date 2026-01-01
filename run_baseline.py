import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from cache_utils import cache_path, load_cache, save_cache
from llm_clients import (
    VISION_PROMPT_VERSION,
    hash_bytes,
    hash_queries,
    hash_text,
    make_cache_key,
    mllm_check,
    mllm_monolithic,
)
from rvad_core import (
    build_sentinel_queries,
    decide_verdict,
    detect_nonvisual_claim,
    extract_claims_regex,
    map_sentinel_results,
    mock_detect_sentinel,
)
from scopes import SCOPE_PROFILES


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class ImageCache:
    """缓存图片读取和 hash 计算,避免重复 I/O"""
    def __init__(self):
        self.cache = {}
    
    def get_image_hash(self, image_path: str) -> str:
        if image_path in self.cache:
            return self.cache[image_path]
        
        if not image_path or not Path(image_path).exists():
            return ""
        
        try:
            image_bytes = Path(image_path).read_bytes()
            img_hash = hash_bytes(image_bytes)
            self.cache[image_path] = img_hash
            return img_hash
        except OSError:
            return ""


BASELINE_B0_PROMPT_VERSION = "b0_monolithic_v1"
BASELINE_B1_PROMPT_VERSION = "b1_monolithic_protocol_v1"


def _default_pred(sample_id, reason="NONVISUAL_UNVERIFIED", evidence_note=""):
    evidence = []
    if evidence_note:
        evidence = [
            {
                "risk_item": "baseline_note",
                "present": "uncertain",
                "bbox": None,
                "score": 0.0,
                "source": "mllm",
                "evidence": evidence_note,
            }
        ]
    return {
        "id": sample_id,
        "verdict": "UNVERIFIED",
        "reason": reason,
        "claims_extracted": [],
        "sentinel_detected": [],
        "evidence": evidence,
    }


def _dedupe_queries(scope_config):
    seen = set()
    out = []
    for config in scope_config.values():
        for q in config.get("queries", []):
            if q in seen:
                continue
            seen.add(q)
            out.append(q)
    return out


def _parse_monolithic_response(sample_id, payload, raw):
    if not isinstance(payload, dict):
        return _default_pred(sample_id, evidence_note="invalid_mllm_payload")
    verdict = payload.get("verdict")
    reason = payload.get("reason")
    free_text = payload.get("free_text", "")
    if verdict not in {"ALERT", "UNVERIFIED", "CONSISTENT"}:
        return _default_pred(sample_id, evidence_note=f"bad_verdict:{verdict}")
    if reason not in {
        "CRITICAL_OMISSION",
        "CONTRADICTION",
        "NONVISUAL_UNVERIFIED",
        "SENTINEL_UNCERTAIN",
        "NONE",
    }:
        return _default_pred(sample_id, evidence_note=f"bad_reason:{reason}")
    evidence = []
    if free_text:
        evidence = [
            {
                "risk_item": "meta_note",
                "present": "uncertain",
                "bbox": None,
                "score": 0.0,
                "source": "baseline",
                "evidence": free_text,
            }
        ]
    if raw:
        evidence.append(
            {
                "risk_item": "meta_raw",
                "present": "uncertain",
                "bbox": None,
                "score": 0.0,
                "source": "baseline",
                "evidence": raw,
            }
        )
    return {
        "id": sample_id,
        "verdict": verdict,
        "reason": reason,
        "claims_extracted": [],
        "sentinel_detected": [],
        "evidence": evidence,
    }


def baseline_predict_pipeline(sample, scope_config, sentinel_detected, sentinel_uncertain, evidence):
    text = sample.get("text", "")
    claims = extract_claims_regex(text, scope_config)
    nonvisual_claim = detect_nonvisual_claim(text)
    verdict, reason = decide_verdict(
        claims, nonvisual_claim, sentinel_detected, sentinel_uncertain
    )
    return {
        "id": sample.get("id"),
        "verdict": verdict,
        "reason": reason,
        "claims_extracted": claims,
        "sentinel_detected": sentinel_detected,
        "evidence": evidence,
    }


def baseline_predict_b2(sample, sentinel_detected, sentinel_uncertain, evidence):
    if sentinel_detected:
        verdict = "ALERT"
        reason = "CRITICAL_OMISSION"
    elif sentinel_uncertain:
        verdict = "UNVERIFIED"
        reason = "SENTINEL_UNCERTAIN"
    else:
        verdict = "CONSISTENT"
        reason = "NONE"
    return {
        "id": sample.get("id"),
        "verdict": verdict,
        "reason": reason,
        "claims_extracted": [],
        "sentinel_detected": sentinel_detected,
        "evidence": evidence,
    }


def process_sample_vision_serial(
    sample: Dict,
    scope_config: Dict,
    claims: List[Dict],
    image_cache: ImageCache,
    mllm_model: str,
    mllm_api_key: str,
    mllm_base_url: str,
    scope_profile: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    args,
) -> Tuple[List[str], List[str], List[Dict], int, bool]:
    """串行处理单个样本的视觉检测 (optimized from important version)"""
    vision_calls = 0
    vision_hit = False
    all_results = {}
    
    image_path = sample.get("image_path", "")
    image_hash = image_cache.get_image_hash(image_path)

    for risk_item, config in scope_config.items():
        queries_for_item = list(config.get("queries", []))
        
        has_absent_claim = any(
            c.get("risk_item") == risk_item and c.get("polarity") == "ASSERT_ABSENT"
            for c in claims
        )
        
        if has_absent_claim:
            extra_queries = config.get("claim_queries", [])
            if extra_queries:
                queries_for_item.extend(extra_queries)
        
        if not queries_for_item:
            continue

        vision_key = make_cache_key({
            "id": sample.get("id"),
            "mllm_model": mllm_model,
            "mllm_base_url": mllm_base_url or "",
            "prompt_version": VISION_PROMPT_VERSION,
            "image_path": image_path,
            "image_hash": image_hash,
            "queries_hash": hash_queries(queries_for_item),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens_vision,
            "seed": args.seed,
            "scope_profile": scope_profile,
            "scope_item_single": risk_item,
        })
        vision_cache_path = cache_path(cache_dir, "vision", vision_key)
        
        if use_cache and vision_cache_path.exists() and not refresh_cache:
            results_for_item = load_cache(vision_cache_path).get("results", {})
            if not vision_hit: 
                vision_hit = True
        else:
            results_for_item, raw_response = mllm_check(
                image_path, queries_for_item, mllm_api_key, mllm_base_url,
                mllm_model, args.temperature, args.max_tokens_vision, args.seed
            )
            vision_calls += 1
            if use_cache:
                save_cache(vision_cache_path, {
                    "results": results_for_item,
                    "raw": raw_response,
                    "meta": {
                        "sample_id": sample.get("id"),
                        "image_path": image_path,
                        "scope_profile": scope_profile,
                        "scope_item": risk_item,
                        "model": mllm_model,
                        "prompt_version": VISION_PROMPT_VERSION,
                    }
                })
        
        all_results.update(results_for_item)

    _, extra_claim_queries = build_sentinel_queries(scope_config, claims)
    sentinel_detected, sentinel_uncertain, evidence = map_sentinel_results(
        scope_config, all_results, extra_claim_queries=extra_claim_queries
    )
    
    return sentinel_detected, sentinel_uncertain, evidence, vision_calls, vision_hit


def main():
    parser = argparse.ArgumentParser(description="Run baseline and write predictions.")
    parser.add_argument("--gt", default="data/gt.jsonl", help="Path to gt jsonl")
    parser.add_argument(
        "--pred", default="outputs/pred_baseline.jsonl", help="Path to output pred jsonl"
    )
    parser.add_argument(
        "--baseline_type",
        choices=[
            "b0_monolithic",
            "b1_monolithic_protocol",
            "b2_scope_scan",
            "ablation_prompted_pipeline",
        ],
        default="b0_monolithic",
    )
    parser.add_argument("--scope_profile", default="nut")
    parser.add_argument("--scope_items", default=None)
    parser.add_argument("--use_real_models", type=int, default=0)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    parser.add_argument("--mllm_model", default=None)
    parser.add_argument("--mllm_base_url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens_vision", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mock_fp_rate", type=float, default=0.0)
    parser.add_argument("--show_progress", type=int, default=1)
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    
    # Ensure output directory exists before starting
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    
    scope_profile = args.scope_profile
    scope_config = SCOPE_PROFILES.get(scope_profile, {})
    if not scope_config:
        print(f"Unknown scope_profile '{scope_profile}', using empty scope.", file=sys.stderr)
    if args.scope_items:
        override_items = [s.strip() for s in args.scope_items.split(",") if s.strip()]
        scope_config = {
            k: scope_config.get(k, {"queries": [], "claim_triggers": []})
            for k in override_items
        }

    use_real_models = bool(int(args.use_real_models))
    use_cache = bool(int(args.use_cache))
    refresh_cache = bool(int(args.refresh_cache))
    cache_dir = Path(args.cache_dir)
    show_progress = bool(args.show_progress)
    mllm_model = args.mllm_model or os.getenv("MLLM_MODEL")
    mllm_base_url = args.mllm_base_url or os.getenv("MLLM_BASE_URL")
    mllm_api_key = os.getenv("MLLM_API_KEY")

    scope_items = list(scope_config.keys())
    monolithic_calls = 0
    vision_calls = 0
    
    # Initialize ImageCache for optimized image hash computation
    image_cache = ImageCache()
    
    # 优化的启动信息输出
    print(f"Baseline: {args.baseline_type} - Streaming Mode", file=sys.stderr)
    print(f"Scope: '{scope_profile}' ({len(scope_items)} items)", file=sys.stderr)
    if use_real_models:
        print(f"Model: vision={mllm_model}", file=sys.stderr)
    else:
        print(f"Model: mock (fp_rate={args.mock_fp_rate})", file=sys.stderr)
    print(f"Cache: {'enabled' if use_cache else 'disabled'}{' (refresh)' if refresh_cache else ''}", file=sys.stderr)
    print(f"Output: {pred_path}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    
    verdict_counts = {}

    with pred_path.open("w", encoding="utf-8") as f:
        # Use enumerate to keep track of the total count
        num_processed = 0
        for i, sample in enumerate(read_jsonl(gt_path)):
            num_processed = i + 1
            sample_id = sample.get("id")
            text = sample.get("text", "")
            image_path = sample.get("image_path", "")
            sample_monolithic_calls = 0
            sample_vision_calls = 0
            text_hit = 0
            vision_hit = 0
            pred = None # Initialize pred

            if args.baseline_type in {"b0_monolithic", "b1_monolithic_protocol"}:
                if not (use_real_models and mllm_model and mllm_api_key):
                    if use_real_models:
                        print("MLLM unavailable; using UNVERIFIED baseline.", file=sys.stderr)
                    pred = _default_pred(sample_id)
                    pred["evidence"].append(
                        {
                            "risk_item": "meta_calls",
                            "present": "uncertain",
                            "bbox": None,
                            "score": 0.0,
                            "source": "meta",
                            "evidence": "monolithic=0,vision=0",
                        }
                    )
                    pred["evidence"].append(
                        {
                            "risk_item": "meta_cache",
                            "present": None,
                            "bbox": None,
                            "score": 0.0,
                            "source": "meta",
                            "evidence": (
                                f"text_hit=0,vision_hit=0,"
                                f"claims_ver=n/a,vision_ver={BASELINE_B0_PROMPT_VERSION if args.baseline_type=='b0_monolithic' else BASELINE_B1_PROMPT_VERSION},"
                                f"scope={scope_profile}"
                            ),
                        }
                    )
                else:
                    prompt_version = (
                        BASELINE_B0_PROMPT_VERSION
                        if args.baseline_type == "b0_monolithic"
                        else BASELINE_B1_PROMPT_VERSION
                    )
                    if args.baseline_type == "b0_monolithic":
                        user_text = (
                            "Given the image and text, output the audit verdict. "
                            "Return JSON as {\"verdict\":\"ALERT|UNVERIFIED|CONSISTENT\", "
                            "\"reason\":\"CRITICAL_OMISSION|CONTRADICTION|NONVISUAL_UNVERIFIED|"
                            "SENTINEL_UNCERTAIN|NONE\", \"free_text\":\"...\"}.\\n"
                            f"Text: {text}"
                        )
                    else:
                        user_text = (
                            "Given the image and text, check each risk item in the scope and "
                            "then output the final audit verdict. "
                            "Return JSON as {\"verdict\":\"ALERT|UNVERIFIED|CONSISTENT\", "
                            "\"reason\":\"CRITICAL_OMISSION|CONTRADICTION|NONVISUAL_UNVERIFIED|"
                            "SENTINEL_UNCERTAIN|NONE\", \"free_text\":\"...\"}.\\n"
                            f"Scope items (JSON array): {json.dumps(scope_items, ensure_ascii=True)}\\n"
                            f"Text: {text}"
                        )

                    image_hash = ""
                    if image_path and Path(image_path).exists():
                        try:
                            image_hash = hash_bytes(Path(image_path).read_bytes())
                        except OSError:
                            image_hash = ""
                    
                    cache_key = make_cache_key(
                        {
                            "id": sample_id,
                            "mllm_model": mllm_model,
                            "mllm_base_url": mllm_base_url or "",
                            "prompt_version": prompt_version,
                            "image_path": image_path,
                            "image_hash": image_hash,
                            "text_hash": hash_text(text),
                            "scope_profile": scope_profile,
                            "scope_items": scope_items,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens_vision,
                            "seed": args.seed,
                        }
                    )
                    cache_path_baseline = cache_path(cache_dir, "baseline", cache_key)
                    if use_cache and cache_path_baseline.exists() and not refresh_cache:
                        cached = load_cache(cache_path_baseline)
                        payload = cached.get("result", {})
                        raw = cached.get("raw", "")
                        vision_hit = 1
                        print(f"cache hit: baseline {sample_id}", file=sys.stderr)
                    else:
                        payload, raw = mllm_monolithic(
                            image_path,
                            user_text,
                            api_key=mllm_api_key,
                            base_url=mllm_base_url,
                            model=mllm_model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens_vision,
                            seed=args.seed,
                        )
                        monolithic_calls += 1
                        sample_monolithic_calls += 1
                        if use_cache:
                            save_cache(
                                cache_path_baseline,
                                {
                                    "result": payload,
                                    "raw": raw,
                                    "meta": {
                                        "sample_id": sample_id,
                                        "image_path": image_path,
                                        "text_preview": text[:120],
                                        "scope_profile": scope_profile,
                                        "scope_items": scope_items,
                                        "model": mllm_model,
                                        "prompt_version": prompt_version,
                                        "temperature": args.temperature,
                                        "max_tokens": args.max_tokens_vision,
                                    },
                                },
                            )
                    pred = _parse_monolithic_response(sample_id, payload, raw)
                    pred["evidence"].append(
                        {
                            "risk_item": "meta_calls",
                            "present": "uncertain",
                            "bbox": None,
                            "score": 0.0,
                            "source": "meta",
                            "evidence": f"monolithic={sample_monolithic_calls},vision=0",
                        }
                    )
                    pred["evidence"].append(
                        {
                            "risk_item": "meta_cache",
                            "present": None,
                            "bbox": None,
                            "score": 0.0,
                            "source": "meta",
                            "evidence": (
                                f"text_hit={text_hit},vision_hit={vision_hit},"
                                f"claims_ver=n/a,vision_ver={prompt_version},"
                                f"scope={scope_profile}"
                            ),
                        }
                    )
            
            elif args.baseline_type == "b2_scope_scan":
                if use_real_models and mllm_model and mllm_api_key:
                    # B2 不需要 claims,传入空列表
                    claims = []
                    
                    # 使用优化的串行视觉处理 (与 ablation 共用相同逻辑)
                    sentinel_detected, sentinel_uncertain, evidence, sample_vision_calls, vision_hit_flag = process_sample_vision_serial(
                        sample, scope_config, claims, image_cache, mllm_model, mllm_api_key, mllm_base_url,
                        scope_profile, cache_dir, use_cache, refresh_cache, args
                    )
                    vision_calls += sample_vision_calls
                    vision_hit = 1 if vision_hit_flag else 0
                else:
                    if use_real_models:
                        print(
                            "MLLM unavailable; baseline falling back to mock detector.",
                            file=sys.stderr,
                        )
                    sentinel_detected = mock_detect_sentinel(
                        fp_rate=args.mock_fp_rate,
                        rng=random.Random(args.seed),
                        scope_items=scope_items,
                    )
                    sentinel_uncertain = []
                    evidence = [
                        {
                            "risk_item": item,
                            "present": True,
                            "bbox": None,
                            "score": 0.0,
                            "source": "mock",
                            "evidence": "mock",
                        }
                        for item in sentinel_detected
                    ]
                    sample_vision_calls = 0
                    vision_hit = 0
                
                pred = baseline_predict_b2(sample, sentinel_detected, sentinel_uncertain, evidence)
                pred["evidence"].append(
                    {
                        "risk_item": "meta_calls",
                        "present": "uncertain",
                        "bbox": None,
                        "score": 0.0,
                        "source": "meta",
                        "evidence": f"monolithic=0,vision={sample_vision_calls}",
                    }
                )
                pred["evidence"].append(
                    {
                        "risk_item": "meta_cache",
                        "present": None,
                        "bbox": None,
                        "score": 0.0,
                        "source": "meta",
                        "evidence": (
                            f"vision_hit={vision_hit},"
                            f"claims_ver=n/a,vision_ver={VISION_PROMPT_VERSION},"
                            f"scope={scope_profile}"
                        ),
                    }
                )
            
            elif args.baseline_type == "ablation_prompted_pipeline":
                if use_real_models and mllm_model and mllm_api_key:
                    claims = extract_claims_regex(text, scope_config)
                    
                    # 使用优化的串行视觉处理
                    sentinel_detected, sentinel_uncertain, evidence, sample_vision_calls, vision_hit_flag = process_sample_vision_serial(
                        sample, scope_config, claims, image_cache, mllm_model, mllm_api_key, mllm_base_url,
                        scope_profile, cache_dir, use_cache, refresh_cache, args
                    )
                    vision_calls += sample_vision_calls
                    vision_hit = 1 if vision_hit_flag else 0
                else:
                    if use_real_models:
                        print(
                            "MLLM unavailable; baseline falling back to mock detector.",
                            file=sys.stderr,
                        )
                    sentinel_detected = mock_detect_sentinel(
                        fp_rate=args.mock_fp_rate,
                        rng=random.Random(args.seed),
                        scope_items=scope_items,
                    )
                    sentinel_uncertain = []
                    evidence = [
                        {
                            "risk_item": item,
                            "present": True,
                            "bbox": None,
                            "score": 0.0,
                            "source": "mock",
                            "evidence": "mock",
                        }
                        for item in sentinel_detected
                    ]
                    sample_vision_calls = 0
                
                pred = baseline_predict_pipeline(
                    sample,
                    scope_config=scope_config,
                    sentinel_detected=sentinel_detected,
                    sentinel_uncertain=sentinel_uncertain,
                    evidence=evidence,
                )
                pred["evidence"].append(
                    {
                        "risk_item": "meta_calls",
                        "present": "uncertain",
                        "bbox": None,
                        "score": 0.0,
                        "source": "meta",
                        "evidence": f"monolithic=0,vision={sample_vision_calls}",
                    }
                )
                pred["evidence"].append(
                    {
                        "risk_item": "meta_cache",
                        "present": None,
                        "bbox": None,
                        "score": 0.0,
                        "source": "meta",
                        "evidence": (
                            f"vision_hit={vision_hit},"
                            f"claims_ver=regex,vision_ver={VISION_PROMPT_VERSION},"
                            f"scope={scope_profile}"
                        ),
                    }
                )
            
            if pred:
                # Write the prediction for the current sample to the file immediately
                f.write(json.dumps(pred, ensure_ascii=True) + "\n")
                f.flush() # Force write to disk
                
                # 统计 verdict
                verdict_counts[pred["verdict"]] = verdict_counts.get(pred["verdict"], 0) + 1
                
                # Optional: Print progress to stderr
                if show_progress:
                    print(f"[{i + 1}] {sample.get('id')}: {pred['verdict']}", file=sys.stderr)

    # Final summary after the loop is complete
    print("\n" + "="*60, file=sys.stderr)
    print(f"✓ Processed {num_processed} samples", file=sys.stderr)
    print(f"✓ Results written to: {pred_path}", file=sys.stderr)
    
    print("\nVerdict Distribution:", file=sys.stderr)
    for verdict, count in sorted(verdict_counts.items()):
        pct = 100 * count / num_processed if num_processed > 0 else 0
        print(f"  {verdict}: {count} ({pct:.1f}%)", file=sys.stderr)
    
    if use_real_models:
        total = monolithic_calls + vision_calls
        print(f"\nMLLM Calls: {total} (monolithic={monolithic_calls}, vision={vision_calls})", file=sys.stderr)
    
    print("="*60, file=sys.stderr)


if __name__ == "__main__":
    main()
