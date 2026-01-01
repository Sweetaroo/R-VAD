import json
from pathlib import Path


def cache_path(cache_dir: Path, kind: str, key: str) -> Path:
    return cache_dir / kind / f"{key}.json"


def load_cache(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
