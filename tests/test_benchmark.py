from pathlib import Path
import pytest

from benchmark import (
    build_prompt_tokens,
    compute_run_metrics,
    format_summary_table,
    load_workloads,
    run_single_request,
    summarize_workload_runs,
)


class FakeTokenizer:
    def __init__(self, bos_token_id=128000):
        self.bos_token_id = bos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        base = [101 + (i % 17) for i in range(len(text.split()))]
        if add_special_tokens and self.bos_token_id is not None:
            return [self.bos_token_id] + base
        return base


def test_load_workloads_matches_spec():
    workloads = load_workloads(Path("workloads/workloads.json"))
    assert [(w["name"], w["prompt_tokens"], w["max_output_tokens"]) for w in workloads] == [
        ("short", 128, 64),
        ("medium", 512, 128),
        ("long", 2048, 256),
    ]


@pytest.mark.parametrize("target_length", [128, 512, 2048])
def test_build_prompt_tokens_exact_length_with_bos(target_length):
    tokenizer = FakeTokenizer(bos_token_id=128000)
    fixture_text = Path("workloads/prompt.txt").read_text(encoding="utf-8")
    tokens = build_prompt_tokens(tokenizer, fixture_text, target_length)

    assert len(tokens) == target_length
    assert tokens[0] == 128000
    assert 128000 not in tokens[1:]
    assert tokens == build_prompt_tokens(tokenizer, fixture_text, target_length)


def test_build_prompt_tokens_without_bos():
    tokenizer = FakeTokenizer(bos_token_id=None)
    tokens = build_prompt_tokens(tokenizer, "alpha beta gamma delta", 10)
    assert len(tokens) == 10


def test_compute_run_metrics_multi_token():
    metrics = compute_run_metrics(
        t0=10.0,
        token_timestamps=[10.050, 10.070, 10.090, 10.110, 10.130],
        t_end=10.135,
        prompt_tokens=128,
        output_tokens=5,
        peak_memory_bytes=2 * (2**30),
    )
    assert metrics["ttft_ms"] == pytest.approx(50.0)
    assert metrics["total_latency_s"] == pytest.approx(0.135)
    # (5 - 1) / (10.130 - 10.050) = 4 / 0.080 = 50.0 tok/s
    assert metrics["decode_tokens_per_sec"] == pytest.approx(50.0)
    assert metrics["prompt_tokens"] == 128
    assert metrics["output_tokens"] == 5
    assert metrics["peak_memory_gib"] == pytest.approx(2.0)


def test_compute_run_metrics_single_token():
    metrics = compute_run_metrics(
        t0=5.0,
        token_timestamps=[5.025],
        t_end=5.030,
        prompt_tokens=512,
        output_tokens=1,
        peak_memory_bytes=1024**3,
    )
    assert metrics["ttft_ms"] == pytest.approx(25.0)
    assert metrics["total_latency_s"] == pytest.approx(0.030)
    assert metrics["decode_tokens_per_sec"] is None
    assert metrics["output_tokens"] == 1
    assert metrics["peak_memory_gib"] == pytest.approx(1.0)


def test_compute_run_metrics_zero_tokens():
    metrics = compute_run_metrics(
        t0=1.0,
        token_timestamps=[],
        t_end=1.015,
        prompt_tokens=128,
        output_tokens=0,
        peak_memory_bytes=1024**3,
    )
    assert metrics["ttft_ms"] is None
    assert metrics["total_latency_s"] == pytest.approx(0.015)
    assert metrics["decode_tokens_per_sec"] is None
    assert metrics["output_tokens"] == 0


def test_compute_run_metrics_zero_decode_elapsed():
    metrics = compute_run_metrics(
        t0=1.0,
        token_timestamps=[1.020, 1.020],
        t_end=1.025,
        prompt_tokens=128,
        output_tokens=2,
        peak_memory_bytes=1024**3,
    )
    assert metrics["decode_tokens_per_sec"] is None


def test_summarize_workload_runs_computes_exact_medians_and_early_stops():
    runs = [
        {
            "workload": "short",
            "repetition": 1,
            "target_prompt_tokens": 128,
            "prompt_tokens": 128,
            "max_output_tokens": 64,
            "output_tokens": 64,
            "stop_reason": "length",
            "early_stop": False,
            "ttft_ms": 20.0,
            "total_latency_s": 0.90,
            "decode_tokens_per_sec": 75.0,
            "peak_memory_gib": 1.85,
        },
        {
            "workload": "short",
            "repetition": 2,
            "target_prompt_tokens": 128,
            "prompt_tokens": 128,
            "max_output_tokens": 64,
            "output_tokens": 42,
            "stop_reason": "stop",
            "early_stop": True,
            "ttft_ms": 18.0,
            "total_latency_s": 0.65,
            "decode_tokens_per_sec": 78.0,
            "peak_memory_gib": 1.86,
        },
        {
            "workload": "short",
            "repetition": 3,
            "target_prompt_tokens": 128,
            "prompt_tokens": 128,
            "max_output_tokens": 64,
            "output_tokens": 64,
            "stop_reason": "length",
            "early_stop": False,
            "ttft_ms": 22.0,
            "total_latency_s": 0.95,
            "decode_tokens_per_sec": 72.0,
            "peak_memory_gib": 1.85,
        },
    ]
    summary = summarize_workload_runs("short", runs)
    assert summary["median_ttft_ms"] == pytest.approx(20.0)
    assert summary["median_total_latency_s"] == pytest.approx(0.90)
    assert summary["median_decode_tokens_per_sec"] == pytest.approx(75.0)
    assert summary["median_peak_memory_gib"] == pytest.approx(1.85)
    assert summary["early_stop_count"] == 1
    assert summary["output_tokens_per_run"] == [64, 42, 64]


def test_run_single_request_stops_on_eos_and_marks_early_stop(monkeypatch):
    import benchmark

    # t0=100.000, t1=100.040, t2=100.060, (EOS breaks before timestamp), t_end=100.085
    clock = iter([100.000, 100.040, 100.060, 100.085])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.mx, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(benchmark.mx, "get_peak_memory", lambda: 2 * (2**30))

    # Yields 2 normal tokens and then EOS (999) before max_tokens=5
    def fake_generate_step(prompt, model, *, max_tokens, **kwargs):
        assert prompt.shape == (128,)
        assert max_tokens == 5
        assert kwargs.get("prompt_cache") is None
        yield 11, None
        yield 12, None
        yield 999, None

    monkeypatch.setattr(benchmark, "generate_step", fake_generate_step)

    class DummyTokenizer:
        eos_token_ids = {999}

    result = run_single_request(
        model=object(),
        tokenizer=DummyTokenizer(),
        prompt_ids=[1] * 128,
        max_output_tokens=5,
        workload_name="short",
        repetition=1,
    )

    assert result["output_tokens"] == 2
    assert result["stop_reason"] == "eos"
    assert result["early_stop"] is True
    assert result["ttft_ms"] == pytest.approx(40.0)
    assert result["decode_tokens_per_sec"] == pytest.approx(50.0)
    assert result["total_latency_s"] == pytest.approx(0.085)
    assert result["peak_memory_gib"] == pytest.approx(2.0)


def test_run_single_request_reaches_cap(monkeypatch):
    import benchmark

    clock = iter([200.000, 200.020, 200.040, 200.060, 200.062])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.mx, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(benchmark.mx, "get_peak_memory", lambda: 2**30)

    def fake_generate_step(prompt, model, *, max_tokens, **kwargs):
        for tok in [10, 20, 30][:max_tokens]:
            yield tok, None

    monkeypatch.setattr(benchmark, "generate_step", fake_generate_step)

    class DummyTokenizer:
        eos_token_ids = {999}

    result = run_single_request(
        model=object(),
        tokenizer=DummyTokenizer(),
        prompt_ids=[1] * 128,
        max_output_tokens=3,
        workload_name="short",
        repetition=1,
    )

    assert result["output_tokens"] == 3
    assert result["stop_reason"] == "length"
    assert result["early_stop"] is False
    assert result["ttft_ms"] == pytest.approx(20.0)
    assert result["decode_tokens_per_sec"] == pytest.approx(50.0)


def test_format_summary_table_includes_medians_and_early_stops():
    summaries = [
        {
            "workload": "short",
            "target_prompt_tokens": 128,
            "prompt_tokens": 128,
            "max_output_tokens": 64,
            "output_tokens_per_run": [64, 42, 64],
            "stop_reasons": ["length", "eos", "length"],
            "early_stop_count": 1,
            "median_ttft_ms": 20.5,
            "median_total_latency_s": 0.85,
            "median_decode_tokens_per_sec": 74.2,
            "median_peak_memory_gib": 1.85,
        }
    ]
    table = format_summary_table(summaries)
    assert "short" in table
    assert "20.50" in table
    assert "0.850" in table
    assert "74.20" in table
    assert "1.850" in table
    assert "1/3" in table
    assert "64,42,64" in table


def test_run_benchmark_saves_nine_runs_and_excludes_warmups(monkeypatch, tmp_path):
    import json
    import benchmark

    def fake_load(model_id_or_path):
        class DummyModel:
            def parameters(self):
                return []

        return (
            DummyModel(),
            FakeTokenizer(bos_token_id=128000),
            {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}},
            Path("/tmp/fake_snapshot/snapshots/0123456789abcdef0123456789abcdef01234567"),
        )

    monkeypatch.setattr(benchmark, "load_mlx_model_and_tokenizer", fake_load)

    call_count = 0

    def fake_run_single_request(
        *, model, tokenizer, prompt_ids, max_output_tokens, workload_name, repetition
    ):
        nonlocal call_count
        call_count += 1
        return {
            "workload": workload_name,
            "repetition": repetition,
            "target_prompt_tokens": len(prompt_ids),
            "prompt_tokens": len(prompt_ids),
            "max_output_tokens": max_output_tokens,
            "output_tokens": max_output_tokens,
            "stop_reason": "length",
            "early_stop": False,
            "ttft_ms": 20.0 + repetition,
            "total_latency_s": 1.0 + 0.1 * repetition,
            "decode_tokens_per_sec": 70.0 + repetition,
            "peak_memory_gib": 1.80,
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

    monkeypatch.setattr(benchmark, "run_single_request", fake_run_single_request)

    out_file = tmp_path / "run.json"
    exit_code = benchmark.main(["--model", "fake/model", "--output", str(out_file)])
    assert exit_code == 0
    # 3 workloads * (1 warmup + 3 measured) = 12 total calls
    assert call_count == 12

    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved["runs"]) == 9
    assert all(r["repetition"] in (1, 2, 3) for r in saved["runs"])
    assert [s["workload"] for s in saved["summaries"]] == ["short", "medium", "long"]
    assert saved["metadata"]["model"]["revision"] == "0123456789abcdef0123456789abcdef01234567"


def test_main_identifies_workload_and_returns_nonzero_on_error(monkeypatch, tmp_path, capsys):
    import benchmark

    bad_workloads = tmp_path / "bad_workloads.json"
    bad_workloads.write_text(
        '{"workloads": [{"name": "broken_workload", "prompt_tokens": 0, "max_output_tokens": 10}]}',
        encoding="utf-8",
    )

    def fake_load(model_id_or_path):
        class DummyModel:
            def parameters(self):
                return []

        return DummyModel(), FakeTokenizer(), {}, Path("/tmp/local")

    monkeypatch.setattr(benchmark, "load_mlx_model_and_tokenizer", fake_load)

    exit_code = benchmark.main(
        [
            "--model",
            "fake/model",
            "--workloads",
            str(bad_workloads),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "broken_workload" in captured.err


def test_saved_hardware_run_json_meets_acceptance_criteria():
    import json
    import statistics

    run_path = Path("results/run.json")
    assert run_path.exists(), "results/run.json must exist from actual hardware run"
    payload = json.loads(run_path.read_text(encoding="utf-8"))

    # Metadata completeness
    meta = payload["metadata"]
    assert meta["model"]["id_or_path"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    assert len(meta["model"]["revision"]) == 40
    assert meta["model"]["quantization"] == {"group_size": 64, "bits": 4}
    assert meta["backend"]["name"] == "mlx-lm"
    assert meta["system"]["chip"] == "Apple M4 Pro"
    assert meta["generation_settings"]["warmup_runs_per_workload"] == 1
    assert meta["generation_settings"]["measured_runs_per_workload"] == 3
    assert meta["memory_measurement"]["label"] == "peak MLX allocated memory"

    # Exactly 9 measured runs, no warmups saved
    runs = payload["runs"]
    assert len(runs) == 9
    assert all(r["repetition"] in (1, 2, 3) for r in runs)

    expected_targets = {"short": (128, 64), "medium": (512, 128), "long": (2048, 256)}
    for r in runs:
        exp_prompt, exp_cap = expected_targets[r["workload"]]
        assert r["target_prompt_tokens"] == exp_prompt
        assert r["prompt_tokens"] == exp_prompt
        assert r["max_output_tokens"] == exp_cap
        assert 0 < r["output_tokens"] <= exp_cap
        assert r["ttft_ms"] > 0
        assert r["total_latency_s"] > 0
        assert r["decode_tokens_per_sec"] > 0
        assert r["peak_memory_gib"] > 0
        assert r["memory_metric_label"] == "peak MLX allocated memory"
        assert "units" in r

    # Verify summary medians match raw runs
    for summary in payload["summaries"]:
        w_name = summary["workload"]
        w_runs = [r for r in runs if r["workload"] == w_name]
        assert len(w_runs) == 3
        assert summary["median_ttft_ms"] == pytest.approx(
            statistics.median(r["ttft_ms"] for r in w_runs)
        )
        assert summary["median_total_latency_s"] == pytest.approx(
            statistics.median(r["total_latency_s"] for r in w_runs)
        )
        assert summary["median_decode_tokens_per_sec"] == pytest.approx(
            statistics.median(r["decode_tokens_per_sec"] for r in w_runs)
        )
        assert summary["median_peak_memory_gib"] == pytest.approx(
            statistics.median(r["peak_memory_gib"] for r in w_runs)
        )



