"""
Label map utilities for mapping disease IDs to human-friendly names.

The label map is expected to be a JSON file whose entries contain at least an
`id` and optional `zh` / `key` fields. The loader is tolerant to a few common
shapes so downstream code can remain stable even if the file layout changes:

- List of objects: [{"id": 0, "key": "AMD", "zh": "年龄相关性黄斑变性"}, ...]
- Dict keyed by id: {"0": {"key": "AMD", "zh": "年龄相关性黄斑变性"}, ...}

Lookups accept numeric ids or strings such as "disease_0"; trailing digits are
used to match the map. Results are cached with an mtime check so the file can
be updated without restarting the app.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


_CACHE: Dict[str, Dict[str, Any]] = {}


def _id_variants(raw_id: Any) -> List[Any]:
    variants: List[Any] = []
    if raw_id is None:
        return variants
    variants.append(raw_id)
    try:
        as_int = int(raw_id)
        variants.append(as_int)
        variants.append(str(as_int))
    except (TypeError, ValueError):
        variants.append(str(raw_id))
    return list(dict.fromkeys(variants))  # deduplicate while preserving order


def _add_entry(mapping: MutableMapping[Any, Dict[str, Any]], raw_id: Any, payload: Mapping[str, Any]):
    zh = payload.get("zh") or payload.get("cn") or payload.get("name_zh")
    key = payload.get("key") or payload.get("en") or payload.get("name") or payload.get("abbr")
    entry = {k: v for k, v in {"zh": zh, "key": key}.items() if v}
    if not entry:
        return
    for vid in _id_variants(raw_id):
        mapping[vid] = entry


def _normalize_labelmap(raw: Any) -> Dict[Any, Dict[str, Any]]:
    mapping: Dict[Any, Dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        labels = raw.get("labels")
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
            for item in labels:
                if isinstance(item, Mapping):
                    _add_entry(mapping, item.get("id"), item)
            return mapping
        if "id" not in raw:
            for rid, payload in raw.items():
                if isinstance(payload, Mapping):
                    _add_entry(mapping, rid, payload)
            return mapping
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                _add_entry(mapping, item.get("id"), item)
    return mapping


def load_labelmap(path: str | os.PathLike[str] | None) -> Dict[Any, Dict[str, Any]]:
    """Load and cache the label map; reloads when the file mtime changes."""
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    try:
        mtime = path_obj.stat().st_mtime
    except OSError:
        return {}

    cache_entry = _CACHE.get(str(path_obj))
    if cache_entry and cache_entry.get("mtime") == mtime:
        return cache_entry["data"]

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    normalized = _normalize_labelmap(raw)
    _CACHE[str(path_obj)] = {"mtime": mtime, "data": normalized}
    return normalized


def resolve_disease_label(disease_id: Any, label_map: Mapping[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """Return {"zh": ..., "key": ...} for the given id (if found)."""
    if not label_map:
        return {}

    candidates: List[Any] = []
    if isinstance(disease_id, str):
        match = re.search(r"(\d+)$", disease_id)
        if match:
            candidates.extend(_id_variants(match.group(1)))
    candidates.extend(_id_variants(disease_id))

    for cid in candidates:
        if cid in label_map:
            return label_map[cid]
    return {}


def enrich_disease_probs(
    disease_probs: Iterable[Mapping[str, Any]], label_map: Mapping[Any, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach zh/key fields to disease prob entries when available."""
    enriched: List[Dict[str, Any]] = []
    for item in disease_probs:
        base = dict(item)
        resolved = resolve_disease_label(item.get("id"), label_map)
        base.update(resolved)
        zh = resolved.get("zh") or base.get("zh")
        key = resolved.get("key") or base.get("key")
        if zh and key:
            base["label"] = f"{zh} ({key})"
        elif zh or key:
            base["label"] = zh or key
        enriched.append(base)
    return enriched
