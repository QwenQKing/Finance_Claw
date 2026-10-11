from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

from .output_schema import RunnerOutput, to_dict
from .logging_utils import append_jsonl, write_json
from .profiling import summarize_stage_timings
from .adapters.simple import build_default_registry, validate_runtime_requirements
from utils.concurrency import resolve_worker_count

def _load_repo_dotenv() -> None:
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.get("extends")
    if not extends:
        return cfg

    base_path = (config_path.parent / str(extends)).resolve()
    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    child_cfg = dict(cfg)
    child_cfg.pop("extends", None)
    return _deep_merge(base_cfg, child_cfg)

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _resolve_seed(cfg: Dict[str, Any], seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    split_seeds = cfg.get("splits", {}).get("seeds", [])
    if isinstance(split_seeds, list) and split_seeds:
        return int(split_seeds[0])
    return 42

def _resolve_max_samples(cfg: Dict[str, Any], max_samples_cli: int | None) -> int | None:
    if max_samples_cli is not None:
        return max_samples_cli

    run_mode = str(cfg.get("run_mode", "")).strip().lower()
    if run_mode == "smoke":
        smoke_cfg = cfg.get("smoke", {}) or {}
        smoke_limit = smoke_cfg.get("max_samples_per_method")
        if smoke_limit is not None:
            return int(smoke_limit)

    generic_limit = cfg.get("max_samples_per_method")
    if generic_limit is not None:
        return int(generic_limit)

    return None

def _resolve_fixed_sample_size(cfg: Dict[str, Any], max_samples: int | None) -> int | None:
    sampling_cfg = cfg.get("sampling", {}) or {}
    fixed_size = sampling_cfg.get("fixed_sample_size")
    if fixed_size is not None and max_samples is not None:
        return min(int(fixed_size), int(max_samples))
    if fixed_size is not None:
        return int(fixed_size)
    return max_samples

def _resolve_fixed_sample_cache_dir(cfg: Dict[str, Any], output_dir: Path) -> Path:
    sampling_cfg = cfg.get("sampling", {}) or {}
    cache_dir = sampling_cfg.get("fixed_sample_cache_dir")
    if cache_dir:
        return Path(cache_dir)
    output_root = cfg.get("output", {}).get("root_dir")
    if output_root:
        return Path(output_root) / "sample_cache"
    return output_dir.parent / "sample_cache"

def _resolve_model_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(cfg.get("model") or cfg.get("llm") or {})
    model_id = (
        str(model_cfg.get("id") or "").strip()
        or os.getenv("MODEL_ID", "").strip()
        or os.getenv("LLM_MODEL_NAME", "").strip()
    )
    model_name = (
        str(model_cfg.get("model_name") or model_cfg.get("name") or "").strip()
        or os.getenv("MODEL_NAME", "").strip()
        or os.getenv("LLM_MODEL_NAME", "").strip()
    )
    base_url = (
        str(model_cfg.get("base_url") or "").strip()
        or os.getenv("LLM_BASE_URL", "").strip()
        or os.getenv("LLM_BASE_URL", "").strip()
    )
    api_key_env = str(model_cfg.get("api_key_env") or os.getenv("LLM_API_KEY_ENV", "") or "LLM_API_KEY").strip()
    api_key_set = bool(os.getenv(api_key_env, "").strip()) if api_key_env else False
    supports_vision = model_cfg.get("supports_vision")
    if supports_vision is None:
        explicit_vision = os.getenv("MODEL_SUPPORTS_VISION", "").strip().lower()
        if explicit_vision:
            supports_vision = explicit_vision in {"1", "true", "yes", "on"}
        else:
            lowered_model = (model_name or model_id).lower()
            supports_vision = any(
                marker in lowered_model
                for marker in ("4o", "vision", "vl", "gpt-5", "gemini", "claude-3", "qwen-vl")
            )
    return {
        "model_id": model_id,
        "model_name": model_name,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key_set": api_key_set,
        "supports_vision": bool(supports_vision),
    }

def _apply_model_runtime_env(model_runtime: Dict[str, Any]) -> None:
    if model_runtime.get("base_url"):
        os.environ["LLM_BASE_URL"] = str(model_runtime["base_url"])
    if model_runtime.get("model_name"):
        os.environ["LLM_MODEL_NAME"] = str(model_runtime["model_name"])
    api_key_env = str(model_runtime.get("api_key_env") or "").strip()
    if api_key_env and api_key_env != "LLM_API_KEY" and os.getenv(api_key_env):
        os.environ["LLM_API_KEY"] = os.getenv(api_key_env, "")
    if model_runtime.get("supports_vision"):
        os.environ["MODEL_SUPPORTS_VISION"] = "1"

def _dataset_identity(dataset_path: Path) -> str:
    return dataset_path.stem

def _normalized_dataset_path(dataset_path: Path) -> str:
    return str(dataset_path.resolve()).replace("\\", "/").lower()

def _dataset_cache_stem(dataset_path: Path) -> str:
    normalized = _normalized_dataset_path(dataset_path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{_dataset_identity(dataset_path)}__{digest}"

def _load_or_create_fixed_sample_indices(
    dataset_path: Path,
    data_len: int,
    requested_size: int,
    seed: int,
    cache_dir: Path,
) -> List[int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = _normalized_dataset_path(dataset_path)
    hashed_cache_path = cache_dir / f"{_dataset_cache_stem(dataset_path)}__n{requested_size}.json"
    legacy_cache_path = cache_dir / f"{_dataset_identity(dataset_path)}__n{requested_size}.json"

    def _load_valid_indices(path: Path) -> List[int] | None:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        indices = payload.get("indices", [])
        if not isinstance(indices, list):
            return None
        if len(indices) != min(requested_size, data_len):
            return None
        try:
            normalized_indices = [int(i) for i in indices]
        except (TypeError, ValueError):
            return None
        if any(i < 0 or i >= data_len for i in normalized_indices):
            return None
        payload_path = str(payload.get("dataset_path", "")).replace("\\", "/").lower()
        if payload_path and payload_path != normalized_path:
            return None
        return normalized_indices

    hashed_indices = _load_valid_indices(hashed_cache_path)
    if hashed_indices is not None:
        return hashed_indices

    legacy_indices = _load_valid_indices(legacy_cache_path)
    if legacy_indices is not None:
        payload = {
            "dataset_path": normalized_path,
            "dataset_name": _dataset_identity(dataset_path),
            "requested_size": int(requested_size),
            "available_size": int(data_len),
            "seed": int(seed),
            "indices": legacy_indices,
        }
        with open(hashed_cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return legacy_indices

    rng = random.Random(seed)
    indices = list(range(data_len))
    rng.shuffle(indices)
    indices = indices[: min(requested_size, data_len)]
    payload = {
        "dataset_path": normalized_path,
        "dataset_name": _dataset_identity(dataset_path),
        "requested_size": int(requested_size),
        "available_size": int(data_len),
        "seed": int(seed),
        "indices": indices,
    }
    with open(hashed_cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return indices

def _stable_sample_id(row: Dict[str, Any]) -> str:
    for key in ("id", "sample_id"):
        if key in row and row[key] is not None:
            s = str(row[key]).strip()
            if s:
                return s
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"anon_{digest}"

def _thread_local_adapter_provider(method_name: str):
    local_state = threading.local()

    def _get_adapter() -> Any:
        adapter = getattr(local_state, "adapter", None)
        if adapter is None:
            instance_tag = f"{method_name}-pid{os.getpid()}-tid{threading.get_ident()}"
            adapter = build_default_registry(instance_tag=instance_tag)[method_name]
            local_state.adapter = adapter
        return adapter

    return _get_adapter

def _load_completed_sample_ids(pred_path: Path) -> set[str]:
    if not pred_path.exists():
        return set()
    done: set[str] = set()
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("sample_id", "")).strip()
            if sid:
                done.add(sid)
    return done

def _load_latest_prediction_rows(pred_path: Path) -> List[Dict[str, Any]]:
    if not pred_path.exists():
        return []
    rows_by_sample_id: Dict[str, Dict[str, Any]] = {}
    anonymous_rows: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("sample_id", "")).strip()
            if sid:
                rows_by_sample_id[sid] = obj
            else:
                anonymous_rows.append(obj)
    return [*rows_by_sample_id.values(), *anonymous_rows]

def _load_active_failure_rows(fail_path: Path, completed_ids: set[str]) -> List[Dict[str, Any]]:
    if not fail_path.exists():
        return []
    rows_by_sample_id: Dict[str, Dict[str, Any]] = {}
    anonymous_rows: List[Dict[str, Any]] = []
    with open(fail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("sample_id", "")).strip()
            if sid:
                if sid in completed_ids:
                    continue
                rows_by_sample_id[sid] = obj
            else:
                anonymous_rows.append(obj)
    return [*rows_by_sample_id.values(), *anonymous_rows]

def _write_jsonl_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _prune_stale_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    latest_rows = _load_latest_prediction_rows(pred_path)
    _write_jsonl_rows(pred_path, latest_rows)
    return latest_rows

def _prune_stale_failures(pred_path: Path, fail_path: Path) -> List[Dict[str, Any]]:
    completed_ids = _load_completed_sample_ids(pred_path)
    active_failures = _load_active_failure_rows(fail_path, completed_ids)
    _write_jsonl_rows(fail_path, active_failures)
    return active_failures

def _aggregate_summary_from_disk(
    pred_path: Path,
    fail_path: Path,
    method_name: str,
    resolved_seed: int,
    method_max_workers: int,
    resume_skipped: int,
) -> Dict[str, Any]:
    latencies: List[float] = []
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    execution_mode_counter: Dict[str, int] = {}
    extras_rows: List[Dict[str, Any]] = []
    prediction_rows = _prune_stale_predictions(pred_path)
    ok_count = len(prediction_rows)
    for obj in prediction_rows:
        latencies.append(float(obj.get("latency_ms", 0.0)))
        prompt_tokens_sum += int(obj.get("prompt_tokens", 0))
        completion_tokens_sum += int(obj.get("completion_tokens", 0))
        total_tokens_sum += int(obj.get("total_tokens", 0))
        extras = obj.get("extras") if isinstance(obj.get("extras"), dict) else {}
        extras_rows.append(extras)
        mode = str(extras.get("execution_mode", "unknown"))
        execution_mode_counter[mode] = execution_mode_counter.get(mode, 0) + 1
    active_failures = _prune_stale_failures(pred_path, fail_path)
    fail_count = len(active_failures)
    total_lines = ok_count + fail_count
    summary: Dict[str, Any] = {
        "total": total_lines,
        "failed": fail_count,
        "completed": ok_count,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "prompt_tokens": prompt_tokens_sum,
        "completion_tokens": completion_tokens_sum,
        "total_tokens": total_tokens_sum,
        "avg_total_tokens_per_sample": (total_tokens_sum / max(1, ok_count)),
        "execution_modes": execution_mode_counter,
        "method": method_name,
        "seed": resolved_seed,
        "max_workers": method_max_workers,
    }
    if resume_skipped:
        summary["resume_skipped"] = resume_skipped
    profiling_summary = summarize_stage_timings(
        extras_rows,
        completed_samples=ok_count,
        total_latency_ms=sum(latencies),
    )
    if profiling_summary:
        summary["profiling"] = profiling_summary
    return summary

def _run_one_sample(adapter_or_factory: Any, row: Dict[str, Any], budget: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    sample_id = _stable_sample_id(row)
    t0 = time.perf_counter()
    try:
        adapter = adapter_or_factory() if callable(adapter_or_factory) else adapter_or_factory
        pred = adapter.run_sample(row, budget=budget)
        latency_ms = (time.perf_counter() - t0) * 1000
        extras = dict(pred.get("extras") or {})
        model_runtime = dict(budget.get("model_runtime") or {})
        if model_runtime:
            extras.setdefault("model_id", model_runtime.get("model_id"))
            extras.setdefault("model_name", model_runtime.get("model_name"))
            extras.setdefault("model_base_url", model_runtime.get("base_url"))
            extras.setdefault("model_supports_vision", model_runtime.get("supports_vision"))
        confidence = pred.get("confidence")
        raw_for_confidence = "\n".join(
            str(value or "")
            for value in (
                extras.get("raw_model_output"),
                extras.get("raw_upstream_output"),
                pred.get("raw_model_output"),
                pred.get("prediction"),
            )
        )
        confidence_match = re.search(
            r"<confidence>\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*</confidence>",
            raw_for_confidence,
            flags=re.I | re.S,
        )
        if confidence_match:
            confidence_raw = max(0.0, min(100.0, float(confidence_match.group(1))))
            confidence = confidence_raw / 100.0
            extras["confidence_raw_0_100"] = confidence_raw
            extras["confidence_source"] = "self_reported_confidence_tag"
        else:
            extras.setdefault("confidence_source", "adapter_reported_or_default")
        mode = str(extras.get("execution_mode", "unknown"))
        output = RunnerOutput(
            sample_id=sample_id,
            method=method_name,
            prediction=pred["prediction"],
            confidence=confidence,
            evidence_ids=pred.get("evidence_ids", []),
            temporal_valid=pred.get("temporal_valid"),
            violation=pred.get("violation"),
            latency_ms=latency_ms,
            steps_used=int(pred.get("steps_used", 0)),
            top_k_used=int(pred.get("top_k_used", 0)),
            prompt_tokens=int(pred.get("prompt_tokens", 0)),
            completion_tokens=int(pred.get("completion_tokens", 0)),
            total_tokens=int(pred.get("total_tokens", 0)),
            extras=extras,
        )
        return {
            "ok": True,
            "sample_id": sample_id,
            "latency_ms": latency_ms,
            "mode": mode,
            "output": output,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "sample_id": sample_id,
            "latency_ms": latency_ms,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

def run(
    config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    asset_dir: str | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    resume: bool = True,
    use_tqdm: bool = True,
) -> None:
    method_name = "financeclaw"

    cfg = _load_config(config_path)
    runner_cfg = cfg.get("runner", {}) or {}

    resolved_seed = _resolve_seed(cfg, seed)
    model_runtime = _resolve_model_runtime(cfg)
    _apply_model_runtime_env(model_runtime)
    budget = dict(cfg.get("budget", {}) or {})
    budget["seed"] = resolved_seed
    budget["model_runtime"] = model_runtime

    if asset_dir:
        os.environ["KB_DIR"] = str(asset_dir)

    data = _load_jsonl(dataset_path)
    data = list(data)

    resolved_max_samples = _resolve_max_samples(cfg, max_samples)
    resolved_fixed_sample_size = _resolve_fixed_sample_size(cfg, resolved_max_samples)
    if resolved_fixed_sample_size is not None:
        cache_dir = _resolve_fixed_sample_cache_dir(cfg, output_dir)
        sampled_indices = _load_or_create_fixed_sample_indices(
            dataset_path=dataset_path,
            data_len=len(data),
            requested_size=resolved_fixed_sample_size,
            seed=resolved_seed,
            cache_dir=cache_dir,
        )
        data = [data[i] for i in sampled_indices]
    elif resolved_max_samples is not None:
        rng = random.Random(resolved_seed)
        rng.shuffle(data)
        data = data[:resolved_max_samples]
    else:
        rng = random.Random(resolved_seed)
        rng.shuffle(data)

    registry = build_default_registry()
    global_max_workers = resolve_worker_count(
        explicit=runner_cfg.get("max_workers"),
        env_var="RUNNER_MAX_WORKERS",
        task="runner_samples",
        upper_bound=len(data) if data else 1,
    )

    runtime_check = validate_runtime_requirements([method_name], cfg)
    if not runtime_check.get("ok", False):
        issue_text = " | ".join(runtime_check.get("issues", []))
        raise RuntimeError(f"Runtime requirements check failed: {issue_text}")

    model_summary = {k: v for k, v in model_runtime.items() if k != "api_key_set"}
    model_summary["api_key_set"] = model_runtime.get("api_key_set", False)
    global_summary: Dict[str, Any] = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "seed": resolved_seed,
        "max_samples": resolved_max_samples,
        "runner_max_workers": global_max_workers,
        "model_runtime": model_summary,
        "runtime_check": runtime_check,
        "methods": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    if method_name not in registry:
        raise ValueError(f"Unknown method: {method_name}")

    adapter_provider = _thread_local_adapter_provider(method_name)
    method_dir = output_dir / method_name
    pred_path = method_dir / "predictions.jsonl"
    fail_path = method_dir / "failures.jsonl"

    if not resume:
        if pred_path.exists():
            pred_path.unlink()
        if fail_path.exists():
            fail_path.unlink()

    completed_ids = _load_completed_sample_ids(pred_path) if resume else set()
    pending_rows = [row for row in data if _stable_sample_id(row) not in completed_ids]
    skipped_resume = len(data) - len(pending_rows)

    grand_total = len(pending_rows)
    method_max_workers = min(
        global_max_workers, max(1, grand_total if grand_total > 0 else len(data))
    )
    print(
        f"[{method_name}] start: dataset_rows={len(data)}, pending={grand_total}, "
        f"resume_skipped={skipped_resume}, seed={resolved_seed}, max_workers={method_max_workers}"
    )

    def build_method_budget() -> Dict[str, Any]:
        method_budget = dict(budget)
        method_budget["method"] = method_name
        method_budget["method_seed"] = int(abs(hash((method_name, resolved_seed))) % (2**31 - 1))
        return method_budget

    method_budget_seed = build_method_budget()
    if pending_rows:
        warmup = getattr(adapter_provider, "warmup", None)
        if callable(warmup):
            warmup_started = time.perf_counter()
            warmup(method_budget_seed)
            warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
            print(f"[{method_name}] warmup complete latency_ms={warmup_ms:.1f}")

    processed = 0
    if grand_total == 0:
        print(f"[{method_name}] resume: no pending samples (all already in {pred_path.name}).")
    elif method_max_workers <= 1:
        row_iter: Any = tqdm(
            pending_rows,
            total=grand_total,
            desc=method_name,
            unit="sample",
            dynamic_ncols=True,
            leave=True,
        ) if use_tqdm else pending_rows
        for row in row_iter:
            processed += 1
            sample_id = _stable_sample_id(row)
            progress_pct = (processed / grand_total * 100.0) if grand_total > 0 else 100.0
            result = _run_one_sample(adapter_provider, row, dict(method_budget_seed), method_name)
            if result["ok"]:
                latency_ms = float(result["latency_ms"])
                mode = str(result.get("mode") or "unknown")
                output = result["output"]
                append_jsonl(pred_path, to_dict(output))
                if use_tqdm:
                    row_iter.set_postfix(
                        id=sample_id[:28] + ("..." if len(sample_id) > 28 else ""),
                        ms=f"{latency_ms:.0f}",
                        tok=output.total_tokens,
                        refresh=False,
                    )
                else:
                    print(
                        f"[{method_name}] {processed}/{grand_total} {progress_pct:.1f}% "
                        f"sample={sample_id} latency_ms={latency_ms:.1f} "
                        f"tokens={output.total_tokens} mode={mode}"
                    )
            else:
                latency_ms = float(result["latency_ms"])
                fail_row = {
                    "sample_id": sample_id,
                    "method": method_name,
                    "error_type": result["error_type"],
                    "error_message": result["error_message"],
                    "latency_ms": latency_ms,
                }
                append_jsonl(fail_path, fail_row)
                if use_tqdm:
                    row_iter.set_postfix(
                        id=sample_id[:24],
                        err=str(result["error_type"])[:20],
                        refresh=False,
                    )
                else:
                    print(
                        f"[{method_name}] {processed}/{grand_total} {progress_pct:.1f}% "
                        f"sample={sample_id} FAILED error={result['error_type']} latency_ms={latency_ms:.1f}"
                    )
    else:
        with ThreadPoolExecutor(max_workers=method_max_workers) as executor:
            futures = [
                executor.submit(_run_one_sample, adapter_provider, row, dict(method_budget_seed), method_name)
                for row in pending_rows
            ]
            future_iter: Any = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=method_name,
                unit="sample",
                dynamic_ncols=True,
                leave=True,
            ) if use_tqdm else as_completed(futures)
            for future in future_iter:
                processed += 1
                progress_pct = (processed / grand_total * 100.0) if grand_total > 0 else 100.0
                result = future.result()
                sample_id = str(result.get("sample_id") or "")
                if result["ok"]:
                    latency_ms = float(result["latency_ms"])
                    mode = str(result.get("mode") or "unknown")
                    output = result["output"]
                    append_jsonl(pred_path, to_dict(output))
                    if use_tqdm:
                        future_iter.set_postfix(
                            id=sample_id[:28] + ("..." if len(sample_id) > 28 else ""),
                            ms=f"{latency_ms:.0f}",
                            tok=output.total_tokens,
                            refresh=False,
                        )
                    else:
                        print(
                            f"[{method_name}] {processed}/{grand_total} {progress_pct:.1f}% "
                            f"sample={sample_id} latency_ms={latency_ms:.1f} "
                            f"tokens={output.total_tokens} mode={mode}"
                        )
                else:
                    latency_ms = float(result["latency_ms"])
                    fail_row = {
                        "sample_id": sample_id,
                        "method": method_name,
                        "error_type": result["error_type"],
                        "error_message": result["error_message"],
                        "latency_ms": latency_ms,
                    }
                    append_jsonl(fail_path, fail_row)
                    if use_tqdm:
                        future_iter.set_postfix(
                            id=sample_id[:24],
                            err=str(result["error_type"])[:20],
                            refresh=False,
                        )
                    else:
                        print(
                            f"[{method_name}] {processed}/{grand_total} {progress_pct:.1f}% "
                            f"sample={sample_id} FAILED error={result['error_type']} latency_ms={latency_ms:.1f}"
                        )

    summary = _aggregate_summary_from_disk(
        pred_path,
        fail_path,
        method_name,
        resolved_seed,
        method_max_workers,
        resume_skipped=skipped_resume if resume else 0,
    )
    write_json(method_dir / "summary.json", summary)
    global_summary["methods"][method_name] = summary

    write_json(output_dir / "summary_all_methods.json", global_summary)

def main() -> None:
    _load_repo_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete existing predictions and rerun all samples (no sample-level skip).",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars; print one line per sample (legacy logging).",
    )
    args = parser.parse_args()

    run(
        config_path=Path(args.config),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        asset_dir=args.asset_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        resume=not args.no_resume,
        use_tqdm=not args.no_tqdm,
    )

if __name__ == "__main__":
    main()
