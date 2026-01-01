import argparse
import json
import os
import random
import sys
from pathlib import Path

from cache_utils import cache_path, load_cache, save_cache
from llm_clients import (
    VISION_PROMPT_VERSION,
    hash_bytes,
    hash_queries,
    make_cache_key,
    mllm_check,
    mllm_monolithic,
)
from rvad_core import (
    detect_nonvisual_claim,
    map_sentinel_results,
    mock_detect_sentinel,
)
from scopes import SCOPE_PROFILES


def read_jsonl(path: Path):
    """流式读取 JSONL 文件"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class ImageCache:
    """缓存图片读取和 hash 计算"""
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
    """默认预测结果"""
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
    """去重查询列表"""
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
    """解析 monolithic baseline 的响应"""
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
                "evidence": raw[:500],  # 截断避免太长
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


def baseline_predict_b2(sample, sentinel_detected, sentinel_uncertain, evidence):
    """B2: 仅基于视觉扫描的预测"""
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


def process_baseline_sample(
    sample,
    baseline_type,
    scope_config,
    scope_items,
    scope_profile,
    image_cache,
    mllm_model,
    mllm_api_key,
    mllm_base_url,
    cache_dir,
    use_cache,
    refresh_cache,
    use_real_models,
    rng,
    args,
):
    """处理单个 baseline 样本"""
    sample_id = sample.get("id")
    text = sample.get("text", "")
    image_path = sample.get("image_path", "")
    
    monolithic_calls = 0
    vision_calls = 0
    text_hit = 0
    vision_hit = 0
    
    pred = None
    
    # B0: Monolithic (无 scope)
    if baseline_type == "b0_monolithic":
        if use_real_models and mllm_model and mllm_api_key:
            image_hash = image_cache.get_image_hash(image_path)
            
            user_text = (
                f"Text on package: {text}\\n\\n"
                "Analyze this product image and text. "
                "Decide if this product is safe for consumers. "
                "Return JSON with: {\"verdict\": \"ALERT|UNVERIFIED|CONSISTENT\", "
                "\"reason\": \"CRITICAL_OMISSION|CONTRADICTION|NONVISUAL_UNVERIFIED|SENTINEL_UNCERTAIN|NONE\", "
                "\"free_text\": \"brief explanation\"}."
            )
            
            mono_key = make_cache_key(
                {
                    "id": sample_id,
                    "mllm_model": mllm_model,
                    "mllm_base_url": mllm_base_url or "",
                    "prompt_version": BASELINE_B0_PROMPT_VERSION,
                    "image_path": image_path,
                    "image_hash": image_hash,
                    "text_hash": hash_bytes(text.encode("utf-8")),
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens_vision,
                    "seed": args.seed,
                }
            )
            mono_cache_path = cache_path(cache_dir, "monolithic", mono_key)
            
            if use_cache and mono_cache_path.exists() and not refresh_cache:
                cached = load_cache(mono_cache_path)
                payload = cached.get("payload", {})
                raw = cached.get("raw", "")
                text_hit = 1
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
                if use_cache:
                    save_cache(
                        mono_cache_path,
                        {
                            "payload": payload,
                            "raw": raw,
                            "meta": {
                                "sample_id": sample_id,
                                "image_path": image_path,
                                "model": mllm_model,
                                "prompt_version": BASELINE_B0_PROMPT_VERSION,
                            },
                        },
                    )
            
            pred = _parse_monolithic_response(sample_id, payload, raw)
        else:
            pred = _default_pred(sample_id)
        
        pred["evidence"].append({
            "risk_item": "meta_calls",
            "present": "uncertain",
            "bbox": None,
            "score": 0.0,
            "source": "meta",
            "evidence": f"monolithic={monolithic_calls},vision=0",
        })
    
    # B1: Monolithic with protocol
    elif baseline_type == "b1_monolithic_protocol":
        if use_real_models and mllm_model and mllm_api_key:
            image_hash = image_cache.get_image_hash(image_path)
            
            scope_desc = ", ".join(scope_items) if scope_items else "allergens"
            user_text = (
                f"Text on package: {text}\\n\\n"
                f"Analyze this product for: {scope_desc}.\\n"
                "Check: 1) Are there explicit absence claims? 2) What is visually present?\\n"
                "Return JSON with: {\"verdict\": \"ALERT|UNVERIFIED|CONSISTENT\", "
                "\"reason\": \"CRITICAL_OMISSION|CONTRADICTION|NONVISUAL_UNVERIFIED|SENTINEL_UNCERTAIN|NONE\", "
                "\"free_text\": \"brief explanation\"}."
            )
            
            mono_key = make_cache_key(
                {
                    "id": sample_id,
                    "mllm_model": mllm_model,
                    "mllm_base_url": mllm_base_url or "",
                    "prompt_version": BASELINE_B1_PROMPT_VERSION,
                    "image_path": image_path,
                    "image_hash": image_hash,
                    "text_hash": hash_bytes(text.encode("utf-8")),
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens_vision,
                    "seed": args.seed,
                    "scope_profile": scope_profile,
                    "scope_items": scope_items,
                }
            )
            mono_cache_path = cache_path(cache_dir, "monolithic", mono_key)
            
            if use_cache and mono_cache_path.exists() and not refresh_cache:
                cached = load_cache(mono_cache_path)
                payload = cached.get("payload", {})
                raw = cached.get("raw", "")
                text_hit = 1
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
                if use_cache:
                    save_cache(
                        mono_cache_path,
                        {
                            "payload": payload,
                            "raw": raw,
                            "meta": {
                                "sample_id": sample_id,
                                "image_path": image_path,
                                "scope_profile": scope_profile,
                                "model": mllm_model,
                                "prompt_version": BASELINE_B1_PROMPT_VERSION,
                            },
                        },
                    )
            
            pred = _parse_monolithic_response(sample_id, payload, raw)
        else:
            pred = _default_pred(sample_id)
        
        pred["evidence"].append({
            "risk_item": "meta_calls",
            "present": "uncertain",
            "bbox": None,
            "score": 0.0,
            "source": "meta",
            "evidence": f"monolithic={monolithic_calls},vision=0",
        })
    
    # B2: Scope scan only
    elif baseline_type == "b2_scope_scan":
        if use_real_models and mllm_model and mllm_api_key:
            image_hash = image_cache.get_image_hash(image_path)
            queries = _dedupe_queries(scope_config)
            
            vision_key = make_cache_key(
                {
                    "id": sample_id,
                    "mllm_model": mllm_model,
                    "mllm_base_url": mllm_base_url or "",
                    "prompt_version": VISION_PROMPT_VERSION,
                    "image_path": image_path,
                    "image_hash": image_hash,
                    "queries_hash": hash_queries(queries),
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens_vision,
                    "seed": args.seed,
                    "scope_profile": scope_profile,
                    "scope_items": scope_items,
                }
            )
            vision_cache_path = cache_path(cache_dir, "vision", vision_key)
            
            if use_cache and vision_cache_path.exists() and not refresh_cache:
                cached = load_cache(vision_cache_path)
                results = cached.get("results", {})
                vision_hit = 1
            else:
                results, raw = mllm_check(
                    image_path,
                    queries,
                    api_key=mllm_api_key,
                    base_url=mllm_base_url,
                    model=mllm_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens_vision,
                    seed=args.seed,
                )
                vision_calls += 1
                if use_cache:
                    save_cache(
                        vision_cache_path,
                        {
                            "results": results,
                            "raw": raw,
                            "meta": {
                                "sample_id": sample_id,
                                "image_path": image_path,
                                "scope_profile": scope_profile,
                                "model": mllm_model,
                                "prompt_version": VISION_PROMPT_VERSION,
                            },
                        },
                    )
            
            sentinel_detected, sentinel_uncertain, evidence = map_sentinel_results(
                scope_config, results, extra_claim_queries={}
            )
        else:
            sentinel_detected = mock_detect_sentinel(
                fp_rate=args.mock_fp_rate, rng=rng, scope_items=scope_items
            )
            sentinel_uncertain = []
            evidence = [{
                "risk_item": item,
                "present": True,
                "bbox": None,
                "score": 0.0,
                "source": "mock",
            } for item in sentinel_detected]
        
        pred = baseline_predict_b2(sample, sentinel_detected, sentinel_uncertain, evidence)
        pred["evidence"].append({
            "risk_item": "meta_calls",
            "present": "uncertain",
            "bbox": None,
            "score": 0.0,
            "source": "meta",
            "evidence": f"monolithic=0,vision={vision_calls}",
        })
    
    
    # 添加缓存元数据
    if pred:
        pred["evidence"].append({
            "risk_item": "meta_cache",
            "present": None,
            "bbox": None,
            "score": 0.0,
            "source": "meta",
            "evidence": (
                f"text_hit={text_hit},vision_hit={vision_hit},"
                f"scope={scope_profile}"
            ),
        })
    
    return pred, monolithic_calls, vision_calls


def main():
    parser = argparse.ArgumentParser(description="Run baseline with streaming output.")
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
    
    # 确保输出目录存在
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置
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
    scope_items = list(scope_config.keys())
    
    use_real_models = bool(int(args.use_real_models))
    use_cache = bool(int(args.use_cache))
    refresh_cache = bool(int(args.refresh_cache))
    cache_dir = Path(args.cache_dir)
    show_progress = bool(args.show_progress)
    
    mllm_model = args.mllm_model or os.getenv("MLLM_MODEL")
    mllm_api_key = os.getenv("MLLM_API_KEY")
    mllm_base_url = args.mllm_base_url or os.getenv("MLLM_BASE_URL")
    
    rng = random.Random(args.seed)
    image_cache = ImageCache()
    
    # 打印配置
    print(f"Baseline: {args.baseline_type} - Streaming Mode", file=sys.stderr)
    print(f"Scope: '{scope_profile}' ({len(scope_items)} items)", file=sys.stderr)
    if use_real_models:
        print(f"Model: vision={mllm_model}", file=sys.stderr)
    print(f"Cache: {'enabled' if use_cache else 'disabled'}", file=sys.stderr)
    print(f"Output: {pred_path}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    
    # 统计
    num_processed = 0
    total_monolithic_calls = 0
    total_vision_calls = 0
    verdict_counts = {}
    
    # 流式处理和写入
    with pred_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(read_jsonl(gt_path), 1):
            pred, mono_calls, vis_calls = process_baseline_sample(
                sample, args.baseline_type, scope_config, scope_items,
                scope_profile, image_cache, mllm_model, mllm_api_key,
                mllm_base_url, cache_dir, use_cache, refresh_cache,
                use_real_models, rng, args
            )
            
            if pred:
                # 立即写入
                f.write(json.dumps(pred, ensure_ascii=True) + "\n")
                f.flush()
                
                # 更新统计
                num_processed += 1
                verdict_counts[pred["verdict"]] = verdict_counts.get(pred["verdict"], 0) + 1
                total_monolithic_calls += mono_calls
                total_vision_calls += vis_calls
                
                # 显示进度
                if show_progress:
                    print(f"[{idx}] {sample.get('id')}: {pred['verdict']}", file=sys.stderr)
    
    # 最终统计
    print("\n" + "="*60, file=sys.stderr)
    print(f"✓ Processed {num_processed} samples", file=sys.stderr)
    print(f"✓ Results written to: {pred_path}", file=sys.stderr)
    print("\nVerdict Distribution:", file=sys.stderr)
    for verdict, count in sorted(verdict_counts.items()):
        pct = 100 * count / num_processed if num_processed > 0 else 0
        print(f"  {verdict}: {count} ({pct:.1f}%)", file=sys.stderr)
    
    if use_real_models:
        total = total_monolithic_calls + total_vision_calls
        print(f"\nMLLM Calls: {total} (monolithic={total_monolithic_calls}, vision={total_vision_calls})", file=sys.stderr)
    
    print("="*60, file=sys.stderr)


if __name__ == "__main__":
    main()