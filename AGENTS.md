# AGENTS.md — InferenceBench

## Purpose & Stack
- **Purpose:** Local CLI benchmark tool for measuring single-stream LLM inference latency, throughput, and peak memory on Apple Silicon (MacBook Pro M4 Pro).
- **Stack:** Python 3.11+ (tested on 3.12), `mlx` and `mlx-lm` (single backend only), `pytest` for unit tests.
- **Default Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (configurable via `--model`).

## Key Files & Responsibilities
- `benchmark.py`: Single-file benchmark implementation (prompt construction, streaming token measurement, MLX memory tracking, summary table, JSON serialization).
- `workloads/workloads.json`: Workload definitions (`short`: 128/64, `medium`: 512/128, `long`: 2048/256).
- `workloads/prompt.txt`: Fixed deterministic text fixture for tokenizing and repeating/truncating to exact prompt lengths.
- `tests/test_benchmark.py`: Unit and evaluation tests (synthetic timestamp calculations, zero/one-token cases, exact prompt-length builders, schema validation).
- `results/`: Output directory for raw JSON benchmark runs.
- `README.md` & `report.md`: Setup guide, measurement boundaries, and hardware benchmark analysis.

## Engineering & Measurement Rules
- **Single Backend:** Use MLX (`mlx-lm`) only. Do not add multi-backend abstractions.
- **Exact Prompt Lengths:** Tokenize `workloads/prompt.txt` without a chat template, include any required BOS token once at the start, repeat/truncate token IDs to exact target length (`128`, `512`, `2048`), and assert length before generation.
- **No State Sharing:** Run one request at a time with fresh KV cache per run (`1` warm-up + `3` measured runs per workload).
- **Synchronize & Timing:** Call `mx.synchronize()` before `t0` and before `t_end` using `time.perf_counter()`.
  - `TTFT (ms) = (t1 - t0) * 1000`
  - `Total latency (s) = t_end - t0`
  - `Decode tokens/sec = (N - 1) / (tN - t1)` (`None` if `N < 2` or `tN <= t1`)
- **Memory:** Call `mx.reset_peak_memory()` before each run and record `mx.get_peak_memory() / (2 ** 30)` GiB labeled as `peak MLX allocated memory`.
- **TDD & Evals:** Write failing deterministic tests in `tests/test_benchmark.py` before implementing or changing logic in `benchmark.py`. Never invent hardware benchmark numbers.

## Commands
- **Install dependencies:** `.venv/bin/pip install -r requirements.txt`
- **Run unit tests:** `.venv/bin/pytest`
- **Run benchmark:** `.venv/bin/python benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --output results/run.json`
