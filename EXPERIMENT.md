# R-VAD 3.1 Experiment Guide

This document describes the experiment goal, current architecture, configuration, and step-by-step execution for the minimal yet reproducible R-VAD 3.1 audit protocol.

## 1. Experiment goal

We evaluate an audit protocol (R-VAD) rather than raw model perception. The pipeline:

1. Extracts explicit safety claims from text.
2. Scans images for critical risk items (scope-driven, tri-state).
3. Applies a deterministic verdict gate with fixed priority.
4. Produces predictions and evaluates UAR / CO-Recall / CR.

Key properties:

- Deterministic cache-based evaluation (no repeated API calls).
- Scope-driven risk items (nut-only or allergens4 and others).
- JSON-only model outputs with strict parsing and fallbacks.
- One-image-one-call MLLM inference with unified tri-state evidence.

## 2. Project layout

- `data/gt.jsonl` Ground-truth data (one JSON per line)
- `data/gt_small.jsonl` Minimal 3-sample sanity set
- `outputs/` Prediction outputs and cache
- `outputs/cache/text/` Text LLM cache
- `outputs/cache/vision/` MLLM cache
- `run_rvad.py` Main R-VAD runner (LLM + MLLM + protocol)
- `run_baseline.py` Baseline runner (protocol or naive)
- `evaluate.py` Metrics and confusion matrix
- `rvad_core.py` Protocol logic and aggregation
- `llm_clients.py` LLM/MLLM API wrappers
- `scopes.py` Scope profiles (risk items, queries, claim triggers)

## 3. Data format

Each line in `data/gt.jsonl` must contain:

```json
{
  "id": "A_001",
  "image_path": "images/A_001.jpg",
  "text": "Premium dark chocolate. Rich cocoa flavor.",
  "gt_label": "ALERT_CRITICAL_OMISSION",
  "gt_critical_present": ["allergen_nut"],
  "gt_explicit_claims": ["allergen_nut:ASSERT_ABSENT"]
}
```

Labels:

- `ALERT_CRITICAL_OMISSION`
- `ALERT_CONTRADICTION`
- `UNVERIFIED`
- `CONSISTENT`

## 4. Output format

`outputs/pred_rvad.jsonl` lines follow this schema:

```json
{
  "id": "A_001",
  "verdict": "ALERT|UNVERIFIED|CONSISTENT",
  "reason": "CONTRADICTION|CRITICAL_OMISSION|NONVISUAL_UNVERIFIED|SENTINEL_UNCERTAIN|NONE",
  "claims_extracted": [
    {"risk_item":"allergen_nut","polarity":"ASSERT_ABSENT","explicit":true,"span":"Nut-free"}
  ],
  "sentinel_detected": ["allergen_nut"],
  "evidence": [
    {"risk_item":"allergen_nut","present":true,"bbox":null,"score":0.9,"source":"mllm","evidence":"visible nuts"}
  ]
}
```

Notes:

- `sentinel_detected` only contains `present==true` items.
- `evidence.present` is tri-state: `true`, `false`, or `"uncertain"`.

## 5. Scope profiles

Scopes are defined in `scopes.py`.

Profiles:

- `nut` (default)
- `allergens4` (nut/seafood/dairy/gluten)
- `allergens4_plus` (same as allergens4, placeholder for future extension)
- `regulated_demo` (cannabis/medication; optional)

You can override the profile using:

```
--scope_profile allergens4
```

Or specify exact items:

```
--scope_items allergen_nut,allergen_seafood
```

## 6. Protocol logic

Fixed verdict priority (do not reorder):

1. `ALERT + CONTRADICTION` if explicit `ASSERT_ABSENT` claim conflicts with `present==true`.
2. `ALERT + CRITICAL_OMISSION` if any `present==true` risk item lacks explicit claim.
3. `UNVERIFIED + SENTINEL_UNCERTAIN` if any risk item is `uncertain` (with confidence >= 0.4).
4. `UNVERIFIED + NONVISUAL_UNVERIFIED` for non-visual claims (e.g., sterile, organic, pesticide-free).
5. `CONSISTENT` otherwise.

## 7. Metrics (definitions and computation)

All metrics are computed from the mapped labels in `evaluate.py`:

Mapping from `(verdict, reason)` to label:

- `(ALERT, CRITICAL_OMISSION)` -> `ALERT_CRITICAL_OMISSION`
- `(ALERT, CONTRADICTION)` -> `ALERT_CONTRADICTION`
- `(UNVERIFIED, NONVISUAL_UNVERIFIED)` -> `UNVERIFIED`
- `(CONSISTENT, NONE)` -> `CONSISTENT`
- `(UNVERIFIED, SENTINEL_UNCERTAIN)` is treated as `UNVERIFIED` for evaluation

Metrics:

- **UAR** (Unsafe Acceptance Rate): among GT labels in `{ALERT_CRITICAL_OMISSION, ALERT_CONTRADICTION, UNVERIFIED}`, the fraction predicted as `CONSISTENT`.
- **CO-Recall**: among GT `ALERT_CRITICAL_OMISSION` samples, the fraction predicted as `ALERT_CRITICAL_OMISSION`.
- **CR** (Contradiction Recall): among GT `ALERT_CONTRADICTION` samples, the fraction predicted as `ALERT_CONTRADICTION`.

Additionally, the evaluation prints:

- **Confusion matrix** across the 4 labels.
- **Per-label sample counts**.
- **Per-risk breakdown** (optional): contradiction and omission hit rates per risk_item.

## 7. Caching and determinism

Caching is enabled by default:

- Text cache key includes model/base_url/prompt_version/text hash/scope/temperature/max_tokens.
- Vision cache key includes model/base_url/prompt_version/image hash/queries hash/temperature/max_tokens.

Cache files include `meta` information for audit/debug but `meta` is **not** part of the cache key.

Flags:

- `--use_cache 1|0` (default 1)
- `--refresh_cache 1|0` (default 0)
- `--cache_dir outputs/cache`

## 8. Environment variables

Required for real models:

- `TEXT_LLM_API_KEY`
- `MLLM_API_KEY`
- `TEXT_LLM_MODEL`
- `MLLM_MODEL`

Optional:

- `TEXT_LLM_BASE_URL`
- `MLLM_BASE_URL`

Initialize quickly:

```bash
source set_env.sh
```

## 9. Step-by-step execution

### 9.1 Sanity check (small set)

```bash
python run_rvad.py --gt data/gt_small.jsonl --pred outputs/pred_rvad_real.jsonl --use_real_models 1 --scope_profile nut
python evaluate.py --gt data/gt_small.jsonl --pred outputs/pred_rvad_real.jsonl
```

### 9.2 Full experiment (real data)

```bash
python run_rvad.py --gt data/gt.jsonl --pred outputs/pred_rvad_real.jsonl --use_real_models 1 --scope_profile allergens4
python evaluate.py --gt data/gt.jsonl --pred outputs/pred_rvad_real.jsonl
```

### 9.3 Cache validation

Run the same command twice; the second run should print cache hits and make no API calls.

```bash
python run_rvad.py --gt data/gt.jsonl --pred outputs/pred_rvad_real.jsonl --use_real_models 1 --scope_profile allergens4
```

To refresh:

```bash
python run_rvad.py --gt data/gt.jsonl --pred outputs/pred_rvad_real.jsonl --use_real_models 1 --scope_profile allergens4 --refresh_cache 1
```

### 9.4 Baselines (B0/B1/B2) and ablation

Main baselines for the primary table:

- **B0 (b0_monolithic)**: one-shot MLLM decides verdict from image+text.
- **B1 (b1_monolithic_protocol)**: one-shot MLLM with scope-aware instructions.
- **B2 (b2_scope_scan)**: scope-only visual scan; no claim reasoning.

Ablation:

- **ablation_prompted_pipeline**: pipeline-like but with regex claims (no text LLM).

```bash
python run_baseline.py --gt data/gt.jsonl --pred outputs/pred_b0.jsonl --baseline_type b0_monolithic --use_real_models 1
python evaluate.py --gt data/gt.jsonl --pred outputs/pred_b0.jsonl

python run_baseline.py --gt data/gt.jsonl --pred outputs/pred_b1.jsonl --baseline_type b1_monolithic_protocol --use_real_models 1 --scope_profile allergens4
python evaluate.py --gt data/gt.jsonl --pred outputs/pred_b1.jsonl

python run_baseline.py --gt data/gt.jsonl --pred outputs/pred_b2.jsonl --baseline_type b2_scope_scan --use_real_models 1 --scope_profile allergens4
python evaluate.py --gt data/gt.jsonl --pred outputs/pred_b2.jsonl

python run_baseline.py --gt data/gt.jsonl --pred outputs/pred_ablation.jsonl --baseline_type ablation_prompted_pipeline --use_real_models 1 --scope_profile allergens4
python evaluate.py --gt data/gt.jsonl --pred outputs/pred_ablation.jsonl
```

## 10. Troubleshooting

- If you see `MLLM returned non-JSON output`, increase `--max_tokens_vision` or reduce evidence verbosity.
- If cache is always missed, verify scope and model settings are unchanged.
- If results are empty, confirm `data/gt.jsonl` is non-empty.
