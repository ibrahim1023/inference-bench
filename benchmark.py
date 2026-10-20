import argparse
import datetime
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.utils import _download, load_model, load_tokenizer

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_WORKLOADS_PATH = Path("workloads/workloads.json")
DEFAULT_PROMPT_PATH = Path("workloads/prompt.txt")
WARMUP_RUNS = 1
MEASURED_RUNS = 3


def load_workloads(workloads_path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(workloads_path).read_text(encoding="utf-8"))
    workloads = data.get("workloads", [])
    if not workloads:
        raise ValueError(f"No workloads found in {workloads_path}")
    return workloads


def build_prompt_tokens(tokenizer: Any, fixture_text: str, target_length: int) -> list[int]:
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    base_ids = list(tokenizer.encode(fixture_text, add_special_tokens=False))
    if not base_ids:
        raise ValueError("Fixture text produced zero tokens")

    bos_id = getattr(tokenizer, "bos_token_id", None)
    prefix: list[int] = [int(bos_id)] if bos_id is not None else []

    needed = target_length - len(prefix)
    if needed > 0:
        repeats = (needed // len(base_ids)) + 1
        body = (base_ids * repeats)[:needed]
        tokens = prefix + body
    else:
        tokens = prefix[:target_length]

    assert len(tokens) == target_length, (
        f"Expected prompt token length {target_length}, got {len(tokens)}"
    )
    return [int(t) for t in tokens]


def compute_run_metrics(
    *,
    t0: float,
    token_timestamps: list[float],
    t_end: float,
    prompt_tokens: int,
    output_tokens: int,
    peak_memory_bytes: int,
) -> dict[str, Optional[float | int]]:
    if output_tokens != len(token_timestamps):
        raise ValueError(
            f"output_tokens ({output_tokens}) does not match timestamp count ({len(token_timestamps)})"
        )
    if peak_memory_bytes is None or peak_memory_bytes < 0:
        raise ValueError("Peak memory bytes could not be measured")

    total_latency_s = t_end - t0
    if output_tokens >= 1:
        t1 = token_timestamps[0]
        ttft_ms: Optional[float] = (t1 - t0) * 1000.0
    else:
        ttft_ms = None

    if output_tokens >= 2:
        t1 = token_timestamps[0]
        t_n = token_timestamps[-1]
        decode_elapsed = t_n - t1
        decode_tokens_per_sec: Optional[float] = (
            (output_tokens - 1) / decode_elapsed if decode_elapsed > 0.0 else None
        )
    else:
        decode_tokens_per_sec = None

    peak_memory_gib = peak_memory_bytes / float(2**30)

    return {
        "ttft_ms": ttft_ms,
        "total_latency_s": total_latency_s,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "peak_memory_gib": peak_memory_gib,
    }


def greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def run_single_request(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_output_tokens: int,
    workload_name: str,
    repetition: int,
) -> dict[str, Any]:
    target_prompt_tokens = len(prompt_ids)
    prompt_array = mx.array(prompt_ids, dtype=mx.int32)
    assert int(prompt_array.size) == target_prompt_tokens, (
        f"Prompt tensor length {prompt_array.size} != {target_prompt_tokens}"
    )

    eos_token_ids = set(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        eos_token_ids.add(int(eos_id))

    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()

    token_timestamps: list[float] = []
    generated_token_ids: list[int] = []
    stop_reason = "length"

    t0 = time.perf_counter()
    step_iter = generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=max_output_tokens,
        sampler=greedy_sampler,
        prompt_cache=None,
    )

    for token_id, _logprobs in step_iter:
        if int(token_id) in eos_token_ids:
            stop_reason = "eos"
            break
        timestamp = time.perf_counter()
        generated_token_ids.append(int(token_id))
        token_timestamps.append(timestamp)
        if len(generated_token_ids) >= max_output_tokens:
            stop_reason = "length"
            break

    mx.synchronize()
    t_end = time.perf_counter()
    peak_memory_bytes = int(mx.get_peak_memory())
    if peak_memory_bytes <= 0:
        raise RuntimeError(
            f"Workload '{workload_name}': failed to measure positive peak MLX allocated memory"
        )

    output_tokens = len(generated_token_ids)
    assert output_tokens <= max_output_tokens, (
        f"Output tokens ({output_tokens}) exceeded max_output_tokens ({max_output_tokens})"
    )
    early_stop = output_tokens < max_output_tokens

    metrics = compute_run_metrics(
        t0=t0,
        token_timestamps=token_timestamps,
        t_end=t_end,
        prompt_tokens=int(prompt_array.size),
        output_tokens=output_tokens,
        peak_memory_bytes=peak_memory_bytes,
    )

    return {
        "workload": workload_name,
        "repetition": repetition,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens": metrics["prompt_tokens"],
        "max_output_tokens": max_output_tokens,
        "output_tokens": metrics["output_tokens"],
        "stop_reason": stop_reason,
        "early_stop": early_stop,
        "ttft_ms": metrics["ttft_ms"],
        "total_latency_s": metrics["total_latency_s"],
        "decode_tokens_per_sec": metrics["decode_tokens_per_sec"],
        "peak_memory_gib": metrics["peak_memory_gib"],
        "memory_metric_label": "peak MLX allocated memory",
        "units": {
            "ttft_ms": "milliseconds",
            "total_latency_s": "seconds",
            "decode_tokens_per_sec": "tokens/second",
            "prompt_tokens": "tokens",
            "output_tokens": "tokens",
            "peak_memory_gib": "GiB (bytes / 2^30)",
        },
    }


def summarize_workload_runs(workload_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError(f"No runs to summarize for workload '{workload_name}'")

    ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
    latencies = [r["total_latency_s"] for r in runs]
    decode_rates = [
        r["decode_tokens_per_sec"] for r in runs if r["decode_tokens_per_sec"] is not None
    ]
    memories = [r["peak_memory_gib"] for r in runs]

    return {
        "workload": workload_name,
        "target_prompt_tokens": runs[0]["target_prompt_tokens"],
        "prompt_tokens": runs[0]["prompt_tokens"],
        "max_output_tokens": runs[0]["max_output_tokens"],
        "output_tokens_per_run": [r["output_tokens"] for r in runs],
        "stop_reasons": [r["stop_reason"] for r in runs],
        "early_stop_count": sum(1 for r in runs if r.get("early_stop")),
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "median_total_latency_s": statistics.median(latencies),
        "median_decode_tokens_per_sec": statistics.median(decode_rates) if decode_rates else None,
        "median_peak_memory_gib": statistics.median(memories),
    }


def format_summary_table(summaries: list[dict[str, Any]]) -> str:
    headers = [
        "Workload",
        "Prompt Tok",
        "Out Tok (runs)",
        "Early Stops",
        "Med TTFT (ms)",
        "Med Latency (s)",
        "Med Decode (tok/s)",
        "Med Peak Mem (GiB)*",
    ]
    rows: list[list[str]] = []
    for s in summaries:
        out_runs = ",".join(str(x) for x in s["output_tokens_per_run"])
        total_runs = len(s["output_tokens_per_run"])
        early_str = f"{s['early_stop_count']}/{total_runs}"
        ttft_str = (
            f"{s['median_ttft_ms']:.2f}" if s["median_ttft_ms"] is not None else "null"
        )
        lat_str = f"{s['median_total_latency_s']:.3f}"
        dec_str = (
            f"{s['median_decode_tokens_per_sec']:.2f}"
            if s["median_decode_tokens_per_sec"] is not None
            else "null"
        )
        mem_str = f"{s['median_peak_memory_gib']:.3f}"
        rows.append(
            [
                str(s["workload"]),
                str(s["prompt_tokens"]),
                f"{out_runs} (cap {s['max_output_tokens']})",
                early_str,
                ttft_str,
                lat_str,
                dec_str,
                mem_str,
            ]
        )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    sep = "-+-".join("-" * w for w in widths)
    lines = [
        fmt_row(headers),
        sep,
        *(fmt_row(r) for r in rows),
        "",
        "* Memory is peak MLX allocated memory (bytes / 2^30 GiB), including resident model weights.",
    ]
    return "\n".join(lines)


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _resolve_model_revision(model_id_or_path: str, resolved_path: Path) -> str:
    match = re.search(r"/snapshots/([0-9a-f]{40})", str(resolved_path))
    if match:
        return match.group(1)
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id_or_path)
        if info.sha:
            return info.sha
    except Exception:
        pass
    return "local"


def _get_available_ram_gib() -> Optional[float]:
    vm_out = _run_cmd(["vm_stat"])
    if vm_out == "unknown":
        return None
    page_size = 16384
    m_ps = re.search(r"page size of (\d+) bytes", vm_out)
    if m_ps:
        page_size = int(m_ps.group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{key}:\s+(\d+)\.", vm_out)
        if m:
            pages += int(m.group(1))
    return round((pages * page_size) / float(2**30), 2) if pages > 0 else None


def collect_environment_metadata(
    *,
    model_id_or_path: str,
    resolved_path: Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    chip = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_bytes_str = _run_cmd(["sysctl", "-n", "hw.memsize"])
    total_ram_gib = (
        round(int(mem_bytes_str) / float(2**30), 2) if mem_bytes_str.isdigit() else None
    )
    macos_version = _run_cmd(["sw_vers", "-productVersion"])
    macos_build = _run_cmd(["sw_vers", "-buildVersion"])
    quantization = model_config.get("quantization") or model_config.get(
        "quantization_config"
    )

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": {
            "id_or_path": model_id_or_path,
            "resolved_path": str(resolved_path),
            "revision": _resolve_model_revision(model_id_or_path, resolved_path),
            "model_type": model_config.get("model_type"),
            "quantization": quantization,
        },
        "backend": {
            "name": "mlx-lm",
            "python_version": platform.python_version(),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "device": str(mx.default_device()),
        },
        "system": {
            "macos_version": macos_version,
            "macos_build": macos_build,
            "chip": chip,
            "total_ram_gib": total_ram_gib,
            "available_ram_gib_at_start": _get_available_ram_gib(),
        },
        "generation_settings": {
            "decoding": "greedy (argmax, temperature=0.0)",
            "streaming": True,
            "batch_size": 1,
            "shared_kv_cache": False,
            "chat_template_used": False,
            "eos_handling": (
                "Stop immediately when token_id is in tokenizer.eos_token_ids; "
                "EOS token itself is not counted in output_tokens."
            ),
            "warmup_runs_per_workload": WARMUP_RUNS,
            "measured_runs_per_workload": MEASURED_RUNS,
        },
        "memory_measurement": {
            "label": "peak MLX allocated memory",
            "source": "mlx.core.get_peak_memory() after mlx.core.reset_peak_memory() per run",
            "scope": "MLX Metal allocator peak allocated bytes (includes resident model weights and KV cache), not total macOS or process RSS memory",
            "unit": "GiB (bytes / 2^30)",
        },
    }


def load_mlx_model_and_tokenizer(model_id_or_path: str) -> tuple[Any, Any, dict[str, Any], Path]:
    resolved_path = _download(model_id_or_path)
    model, config = load_model(resolved_path, lazy=False)
    tokenizer = load_tokenizer(
        resolved_path,
        {"trust_remote_code": False},
        eos_token_ids=config.get("eos_token_id", None),
    )
    mx.eval(model.parameters())
    mx.synchronize()
    return model, tokenizer, config, resolved_path


def run_benchmark(
    *,
    model_id_or_path: str,
    output_path: Path,
    workloads_path: Path = DEFAULT_WORKLOADS_PATH,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> dict[str, Any]:
    workloads = load_workloads(workloads_path)
    fixture_text = Path(prompt_path).read_text(encoding="utf-8")

    print(f"Loading model '{model_id_or_path}' with MLX...")
    model, tokenizer, config, resolved_path = load_mlx_model_and_tokenizer(model_id_or_path)

    prepared_prompts: dict[str, list[int]] = {}
    for w in workloads:
        w_name = w["name"]
        try:
            target_len = int(w["prompt_tokens"])
            tokens = build_prompt_tokens(tokenizer, fixture_text, target_len)
            assert len(tokens) == target_len
            prepared_prompts[w_name] = tokens
        except Exception as exc:
            raise RuntimeError(f"Workload '{w_name}' prompt preparation failed: {exc}") from exc

    metadata = collect_environment_metadata(
        model_id_or_path=model_id_or_path,
        resolved_path=resolved_path,
        model_config=config,
    )

    all_measured_runs: list[dict[str, Any]] = []
    workload_summaries: list[dict[str, Any]] = []

    for w in workloads:
        w_name = w["name"]
        prompt_ids = prepared_prompts[w_name]
        max_output_tokens = int(w["max_output_tokens"])
        print(
            f"Running workload '{w_name}' (prompt_tokens={len(prompt_ids)}, "
            f"max_output_tokens={max_output_tokens}): {WARMUP_RUNS} warm-up + {MEASURED_RUNS} measured..."
        )
        try:
            for warmup_idx in range(WARMUP_RUNS):
                _ = run_single_request(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_output_tokens=max_output_tokens,
                    workload_name=w_name,
                    repetition=-(warmup_idx + 1),
                )

            workload_runs: list[dict[str, Any]] = []
            for rep in range(1, MEASURED_RUNS + 1):
                run_record = run_single_request(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_output_tokens=max_output_tokens,
                    workload_name=w_name,
                    repetition=rep,
                )
                workload_runs.append(run_record)
                all_measured_runs.append(run_record)

            summary = summarize_workload_runs(w_name, workload_runs)
            workload_summaries.append(summary)
        except Exception as exc:
            raise RuntimeError(f"Workload '{w_name}' execution failed: {exc}") from exc

    payload = {
        "metadata": metadata,
        "workloads": workloads,
        "summaries": workload_summaries,
        "runs": all_measured_runs,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + format_summary_table(workload_summaries))
    print(f"\nSaved {len(all_measured_runs)} measured runs to {output_path}")
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local single-stream LLM inference benchmark on Apple Silicon (MLX)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/run.json"),
        help="Path to write output JSON (default: results/run.json)",
    )
    parser.add_argument(
        "--workloads",
        type=Path,
        default=DEFAULT_WORKLOADS_PATH,
        help=f"Path to workloads JSON file (default: {DEFAULT_WORKLOADS_PATH})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"Path to deterministic text fixture (default: {DEFAULT_PROMPT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_benchmark(
            model_id_or_path=args.model,
            output_path=args.output,
            workloads_path=args.workloads,
            prompt_path=args.prompt_file,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
