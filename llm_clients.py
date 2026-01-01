import base64
import hashlib
import json
import mimetypes
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://api.openai.com"
TEXT_PROMPT_VERSION = "claims_v3"
VISION_PROMPT_VERSION = "sentinel_v3"


def _post_json(url, payload, api_key):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP Error {exc.code}: {exc.reason}", file=sys.stderr)
        try:
            error_body = exc.read().decode("utf-8")
            print(f"Error response: {error_body}", file=sys.stderr)
        except:
            pass
        return None
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return None


def _extract_json(text):
    if not text:
        return None
    # Try to extract JSON from markdown code blocks first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    
    # Now try to parse JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def make_cache_key(payload):
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_queries(queries):
    normalized = sorted(queries)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def extract_claims_with_text_llm(
    text,
    api_key,
    base_url=None,
    model=None,
    temperature=0,
    max_tokens=400,
    seed=None,
    scope_items=None,
    scope_definitions=None,
):
    if not api_key or not model:
        return [], ""
    
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    
    if "bigmodel.cn" in base_url or base_url.endswith("/v4"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    scope_items = scope_items or []
    scope_definitions = scope_definitions or {}

    # 简化 system prompt，提高兼容性
    system_prompt = (
        "You are a JSON extractor. Return ONLY valid JSON with no markdown. "
        "Format: {\"claims\": [...]} where each claim has: "
        "risk_item (string), polarity (\"ASSERT_ABSENT\"), explicit (true)."
    )
    
    scope_examples = []
    for item in scope_items[:3]: 
        examples = scope_definitions.get(item, [])
        if examples:
            scope_examples.append(f"{item}: {', '.join(examples[:2])}")
    
    scope_desc = "; ".join(scope_examples)
    if len(scope_items) > 3:
        scope_desc += f" (and {len(scope_items)-3} more)"
    
    user_prompt = (
        "Extract explicit safety claims that state absence (e.g., 'nut-free', 'no dairy'). "
        f"Valid risk items: {', '.join(scope_items)}. "
        "Return JSON: {\"claims\": []} with risk_item, polarity='ASSERT_ABSENT', explicit=true. "
        f"Examples: {scope_desc}\\n\\n"
        f"Text to analyze: {text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if "bigmodel.cn" not in base_url:
        payload["response_format"] = {"type": "json_object"}
    
    # seed 参数可能不被所有 API 支持
    if seed is not None and "bigmodel.cn" not in base_url:
        payload["seed"] = seed

    response = _post_json(url, payload, api_key)
    if not response:
        return [], ""

    content = (
        response.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    
    # 打印原始响应用于调试
    if not content:
        print(f"Empty response from text LLM", file=sys.stderr)
        return [], ""
    
    parsed = _extract_json(content)
    if not parsed:
        print(f"Text LLM returned non-JSON output: {content[:200]}", file=sys.stderr)
        return [], content

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        return [], content

    cleaned = []
    for item in claims:
        if not isinstance(item, dict):
            continue
        if item.get("risk_item") not in scope_items:
            continue
        if item.get("polarity") != "ASSERT_ABSENT":
            continue
        cleaned.append(
            {
                "risk_item": item.get("risk_item"),
                "polarity": "ASSERT_ABSENT",
                "explicit": True,
            }
        )
    return cleaned, content


def _default_mllm_result(queries, evidence_msg=""):
    return {
        q: {
            "present": "uncertain",
            "confidence": 0.0,
            "bbox": None,
            "evidence": evidence_msg,
        }
        for q in queries
    }


def mllm_check(
    image_path,
    queries,
    api_key,
    base_url=None,
    model=None,
    temperature=0,
    max_tokens=600,
    seed=None,
):
    if not api_key or not model:
        return _default_mllm_result(queries), ""

    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    if "bigmodel.cn" in base_url or base_url.endswith("/v4"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except OSError as exc:
        print(f"Failed to read image {image_path}: {exc}", file=sys.stderr)
        return _default_mllm_result(queries, f"image_read_error: {exc}"), ""

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    system_prompt = (
        "You are a visual inspector. Return ONLY valid JSON, no markdown. "
        "Output format: {\"query1\": {\"present\": true/false/\"uncertain\", "
        "\"confidence\": 0-1, \"evidence\": \"brief text\", \"bbox\": null}, ...}"
    )

    user_text = (
        "For each query below, check if it's visible in the image.\\n"
        "Rules:\\n"
        "- present=true: clear visual evidence (confidence >=0.7)\\n"
        "- present=\"uncertain\": ambiguous/occluded/unclear (confidence <=0.4)\\n"
        "- present=false: confidently absent\\n"
        "Return JSON with keys matching queries exactly.\\n\\n"
        f"Queries: {json.dumps(queries)}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    response = _post_json(url, payload, api_key)
    if not response:
        return _default_mllm_result(queries), ""

    content = (
        response.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        print(f"MLLM returned non-JSON: {content[:200]}", file=sys.stderr)
        return _default_mllm_result(queries), content

    missing_keys = [q for q in queries if q not in parsed]
    if missing_keys:
        print(
            f"MLLM response missing {len(missing_keys)} query keys; filling as uncertain.",
            file=sys.stderr,
        )

    results = _default_mllm_result(queries)
    for q in queries:
        item = parsed.get(q)
        if not isinstance(item, dict):
            continue
        present = item.get("present")
        if present not in (True, False, "uncertain"):
            present = "uncertain"
        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = item.get("evidence", "")
        results[q] = {
            "present": present,
            "confidence": max(0.0, min(1.0, confidence)),
            "bbox": None,
            "evidence": evidence if isinstance(evidence, str) else "",
        }

    return results, content


def mllm_monolithic(
    image_path,
    user_text,
    api_key,
    base_url=None,
    model=None,
    temperature=0,
    max_tokens=600,
    seed=None,
):
    if not api_key or not model:
        return {}, ""

    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if "bigmodel.cn" in base_url or base_url.endswith("/v4"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except OSError as exc:
        print(f"Failed to read image {image_path}: {exc}", file=sys.stderr)
        return {}, ""

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    system_prompt = (
        "You are a strict visual-text judge. "
        "Return ONLY valid JSON. No markdown, no explanation, no extra text. "
        "Output must be a single JSON object."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    response = _post_json(url, payload, api_key)
    if not response:
        return {}, ""

    content = (
        response.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        print("MLLM returned non-JSON output for monolithic call.", file=sys.stderr)
        return {}, content

    return parsed, content