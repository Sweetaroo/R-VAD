import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from cache_utils import cache_path, load_cache, save_cache
from llm_clients import (
    TEXT_PROMPT_VERSION,
    VISION_PROMPT_VERSION,
    extract_claims_with_text_llm,
    hash_bytes,
    hash_queries,
    hash_text,
    make_cache_key,
)
from rvad_core import (
    build_sentinel_queries,
    decide_verdict,
    dedupe_claims,
    detect_nonvisual_claim,
    extract_claims_regex,
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


def process_text_claims(
    sample: Dict,
    text_model: str,
    text_api_key: str,
    text_base_url: str,
    scope_items: List[str],
    scope_definitions: Dict,
    scope_profile: str,
    scope_def_hash: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    args,
    use_real_models: bool,
    scope_config: Dict,
) -> Tuple[List[Dict], bool, int, bool]:
    """处理文本 claim 提取"""
    text = sample.get("text", "")
    text_calls = 0
    text_hit = False
    
    if use_real_models and text_model and text_api_key:
        text_key = make_cache_key(
            {
                "id": sample.get("id"),
                "text_model": text_model,
                "text_base_url": text_base_url or "",
                "prompt_version": TEXT_PROMPT_VERSION,
                "text_hash": hash_text(text),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens_text,
                "seed": args.seed,
                "scope_profile": scope_profile,
                "scope_items": scope_items,
                "scope_def_hash": scope_def_hash,
            }
        )
        text_cache_path = cache_path(cache_dir, "text", text_key)
        
        if use_cache and text_cache_path.exists() and not refresh_cache:
            cached = load_cache(text_cache_path)
            claims = cached.get("claims", [])
            text_hit = True
        else:
            claims, raw = extract_claims_with_text_llm(
                text,
                api_key=text_api_key,
                base_url=text_base_url,
                model=text_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens_text,
                seed=args.seed,
                scope_items=scope_items,
                scope_definitions=scope_definitions,
            )
            text_calls += 1
            if use_cache:
                save_cache(
                    text_cache_path,
                    {
                        "claims": claims,
                        "raw": raw,
                        "meta": {
                            "sample_id": sample.get("id"),
                            "text_preview": text[:120],
                            "scope_profile": scope_profile,
                            "scope_items": scope_items,
                            "model": text_model,
                            "prompt_version": TEXT_PROMPT_VERSION,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens_text,
                        },
                    },
                )
    else:
        claims = extract_claims_regex(text, scope_config)
    
    nonvisual_claim = detect_nonvisual_claim(text)
    claims_deduped = dedupe_claims(claims)
    
    return claims_deduped, nonvisual_claim, text_calls, text_hit


def process_vision_for_risk_item(
    risk_item: str,
    config: Dict,
    claims_deduped: List[Dict],
    sample: Dict,
    image_hash: str,
    mllm_model: str,
    mllm_api_key: str,
    mllm_base_url: str,
    scope_profile: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    args,
) -> Tuple[str, Dict, int, bool]:
    """处理单个 risk_item 的视觉检测"""
    queries_for_item = list(config.get("queries", []))
    
    # 检查是否有相关的 ASSERT_ABSENT claim
    has_absent_claim = any(
        c.get("risk_item") == risk_item and c.get("polarity") == "ASSERT_ABSENT"
        for c in claims_deduped
    )
    
    if has_absent_claim:
        extra_queries = config.get("claim_queries", [])
        if extra_queries:
            queries_for_item.extend(extra_queries)
    
    if not queries_for_item:
        return risk_item, {}, 0, False
    
    image_path = sample.get("image_path", "")
    
    # 创建缓存 key
    vision_key = make_cache_key(
        {
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
        }
    )
    vision_cache_path = cache_path(cache_dir, "vision", vision_key)
    
    # 检查缓存
    if use_cache and vision_cache_path.exists() and not refresh_cache:
        cached = load_cache(vision_cache_path)
        results_for_item = cached.get("results", {})
        return risk_item, results_for_item, 0, True
    
    # 调用 MLLM
    from llm_clients import mllm_check
    
    results_for_item, raw = mllm_check(
        image_path,
        queries_for_item,
        api_key=mllm_api_key,
        base_url=mllm_base_url,
        model=mllm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens_vision,
        seed=args.seed,
    )
    
    if use_cache:
        save_cache(
            vision_cache_path,
            {
                "results": results_for_item,
                "raw": raw,
                "meta": {
                    "sample_id": sample.get("id"),
                    "image_path": image_path,
                    "scope_profile": scope_profile,
                    "scope_items": [risk_item],
                    "model": mllm_model,
                    "prompt_version": VISION_PROMPT_VERSION,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens_vision,
                },
            },
        )
    
    return risk_item, results_for_item, 1, False


def process_sample(
    sample: Dict,
    scope_config: Dict,
    scope_items: List[str],
    scope_definitions: Dict,
    scope_profile: str,
    scope_def_hash: str,
    image_cache: ImageCache,
    text_model: str,
    text_api_key: str,
    text_base_url: str,
    mllm_model: str,
    mllm_api_key: str,
    mllm_base_url: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    use_real_models: bool,
    rng: random.Random,
    args,
) -> Tuple[Dict, int, int]:
    """处理单个样本并返回预测结果"""
    
    # 1. 处理文本 claims
    claims_deduped, nonvisual_claim, text_calls, text_hit = process_text_claims(
        sample, text_model, text_api_key, text_base_url,
        scope_items, scope_definitions, scope_profile, scope_def_hash,
        cache_dir, use_cache, refresh_cache, args, use_real_models, scope_config
    )
    
    # 2. 处理视觉检测
    vision_calls = 0
    vision_hit = False
    
    if use_real_models and mllm_model and mllm_api_key:
        # 预先获取图片 hash (只读取一次)
        image_path = sample.get("image_path", "")
        image_hash = image_cache.get_image_hash(image_path)
        
        # 串行处理每个 risk_item
        all_results = {}
        for risk_item, config in scope_config.items():
            _, results, calls, hit = process_vision_for_risk_item(
                risk_item, config, claims_deduped, sample, image_hash,
                mllm_model, mllm_api_key, mllm_base_url, scope_profile,
                cache_dir, use_cache, refresh_cache, args
            )
            all_results.update(results)
            vision_calls += calls
            if hit:
                vision_hit = True
        
        # 生成 sentinel 结果
        _, extra_claim_queries = build_sentinel_queries(scope_config, claims_deduped)
        sentinel_detected, sentinel_uncertain, evidence = map_sentinel_results(
            scope_config, all_results, extra_claim_queries=extra_claim_queries
        )
    else:
        sentinel_detected = mock_detect_sentinel(
            fp_rate=args.mock_fp_rate, rng=rng, scope_items=scope_items
        )
        sentinel_uncertain = []
        evidence = [
            {
                "risk_item": item,
                "present": True,
                "bbox": None,
                "score": 0.0,
                "source": "mock",
            }
            for item in sentinel_detected
        ]
    
    # 3. 决定 verdict
    verdict, reason = decide_verdict(
        claims_deduped, nonvisual_claim, sentinel_detected, sentinel_uncertain
    )
    
    # 4. 构建预测结果
    pred = {
        "id": sample.get("id"),
        "verdict": verdict,
        "reason": reason,
        "claims_extracted": claims_deduped,
        "sentinel_detected": sentinel_detected,
        "evidence": evidence
        + [
            {
                "risk_item": "meta_calls",
                "present": "uncertain",
                "bbox": None,
                "score": 0.0,
                "source": "meta",
                "evidence": f"text={text_calls},vision={vision_calls}",
            },
            {
                "risk_item": "meta_cache",
                "present": None,
                "bbox": None,
                "score": 0.0,
                "source": "meta",
                "evidence": (
                    f"text_hit={int(text_hit)},vision_hit={int(vision_hit)},"
                    f"claims_ver={TEXT_PROMPT_VERSION},"
                    f"vision_ver={VISION_PROMPT_VERSION},"
                    f"scope={scope_profile}"
                ),
            },
        ],
    }
    
    return pred, text_calls, vision_calls


def main():
    parser = argparse.ArgumentParser(description="Run R-VAD with streaming output.")
    parser.add_argument("--gt", default="data/gt.jsonl", help="Path to gt jsonl")
    parser.add_argument(
        "--pred", default="outputs/pred_rvad.jsonl", help="Path to output pred jsonl"
    )
    parser.add_argument("--use_real_models", type=int, default=1)
    parser.add_argument("--mock_fp_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--text_model", default=None)
    parser.add_argument("--mllm_model", default=None)
    parser.add_argument("--text_base_url", default=None)
    parser.add_argument("--mllm_base_url", default=None)
    parser.add_argument("--scope_profile", default="nut")
    parser.add_argument("--scope_items", default=None)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens_text", type=int, default=400)
    parser.add_argument("--max_tokens_vision", type=int, default=600)
    parser.add_argument("--show_progress", type=int, default=1,
                       help="Show progress during processing")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    
    # 确保输出目录存在
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    verdict_counts = {}
    rng = random.Random(args.seed)
    use_real_models = bool(int(args.use_real_models))
    use_cache = bool(int(args.use_cache))
    refresh_cache = bool(int(args.refresh_cache))
    cache_dir = Path(args.cache_dir)
    total_text_calls = 0
    total_vision_calls = 0
    show_progress = bool(args.show_progress)

    # 环境变量
    text_model = args.text_model or os.getenv("TEXT_LLM_MODEL")
    mllm_model = args.mllm_model or os.getenv("MLLM_MODEL")
    text_base_url = args.text_base_url or os.getenv("TEXT_LLM_BASE_URL")
    mllm_base_url = args.mllm_base_url or os.getenv("MLLM_BASE_URL")
    text_api_key = os.getenv("TEXT_LLM_API_KEY")
    mllm_api_key = os.getenv("MLLM_API_KEY")

    # Scope 配置 (只计算一次)
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
    scope_definitions = {
        k: scope_config.get(k, {}).get("claim_triggers", []) for k in scope_items
    }
    scope_def_hash = make_cache_key(scope_definitions)[:16]
    
    # 图片缓存
    image_cache = ImageCache()
    
    # 打印配置信息
    print(f"R-VAD Pipeline - Streaming Mode", file=sys.stderr)
    print(f"Scope: '{scope_profile}' ({len(scope_items)} items)", file=sys.stderr)
    if use_real_models:
        print(f"Models: text={text_model}, vision={mllm_model}", file=sys.stderr)
    print(f"Cache: {'enabled' if use_cache else 'disabled'}, refresh: {refresh_cache}", file=sys.stderr)
    print(f"Output: {pred_path}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    
    # 流式处理和写入
    num_processed = 0
    with pred_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(read_jsonl(gt_path), 1):
            # 处理样本
            pred, text_calls, vision_calls = process_sample(
                sample, scope_config, scope_items, scope_definitions,
                scope_profile, scope_def_hash, image_cache,
                text_model, text_api_key, text_base_url,
                mllm_model, mllm_api_key, mllm_base_url,
                cache_dir, use_cache, refresh_cache, use_real_models, rng, args
            )
            
            # 立即写入文件
            f.write(json.dumps(pred, ensure_ascii=True) + "\n")
            f.flush()  # 强制刷新到磁盘
            
            # 更新统计
            num_processed += 1
            verdict_counts[pred["verdict"]] = verdict_counts.get(pred["verdict"], 0) + 1
            total_text_calls += text_calls
            total_vision_calls += vision_calls
            
            # 显示进度
            if show_progress:
                print(f"[{idx}] {sample.get('id')}: {pred['verdict']}", file=sys.stderr)

    # 打印最终统计
    print("\n" + "="*60, file=sys.stderr)
    print(f"✓ Processed {num_processed} samples", file=sys.stderr)
    print(f"✓ Results written to: {pred_path}", file=sys.stderr)
    print("\nVerdict Distribution:", file=sys.stderr)
    for verdict, count in sorted(verdict_counts.items()):
        pct = 100 * count / num_processed if num_processed > 0 else 0
        print(f"  {verdict}: {count} ({pct:.1f}%)", file=sys.stderr)
    
    if use_real_models:
        total_calls = total_text_calls + total_vision_calls
        print(f"\nAPI Calls: {total_calls} (text={total_text_calls}, vision={total_vision_calls})", file=sys.stderr)
        if use_cache and num_processed > 0:
            expected_calls = num_processed * (1 + len(scope_items))
            cache_saved = expected_calls - total_calls
            print(f"Cache hits saved ~{cache_saved} API calls", file=sys.stderr)
    
    print("="*60, file=sys.stderr)


if __name__ == "__main__":
    main()