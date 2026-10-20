# InferenceBench

A lightweight command-line benchmark tool for measuring single-stream local LLM inference latency, decode throughput, and peak MLX memory on Apple Silicon (`Apple M4 Pro`).

## Purpose

InferenceBench measures the two distinct phases of autoregressive transformer inference—**prefill** (captured in Time to First Token, TTFT) and **decode** (steady-state token generation throughput)—alongside **peak MLX allocated memory** across three deterministic prompt/output workloads (`short`, `medium`, `long`).

It uses a single backend (`mlx-lm` on Apple Silicon Metal), greedy decoding (`argmax`), streaming token generation, and fresh prompt/KV cache state on every request.

## Prerequisites & comparability recommendations

- **Hardware:** Mac with Apple Silicon (benchmarked on MacBook Pro with Apple M4 Pro, 24 GiB unified memory).
- **OS & Python:** macOS with Python 3.11+ (tested with Python 3.12.11).
- **For comparable runs:** Plug the Mac into wall power, disable Low Power Mode, and close heavy background applications (browsers, IDEs, video editors, or other local model servers) so unified memory bandwidth and GPU cores are not contested.

## Environment setup

Create a virtual environment and install the pinned dependencies from `requirements.txt`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model download & configuration

By default, `benchmark.py` uses `mlx-community/Llama-3.2-3B-Instruct-4bit` (revision `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e`, 4-bit quantized with group size 64, ~1.8 GiB weights).

When you run `benchmark.py`, `mlx-lm` automatically downloads the model snapshot to your local Hugging Face cache (`~/.cache/huggingface/hub/`) if it is not already present, or you can pass a local directory path via `--model`. Model download and weight loading occur **before** any benchmark timers start.

## Running the benchmark

Run all three workloads (`short`, `medium`, `long`) with 1 unmeasured warm-up run and 3 measured runs per workload:

```sh
python benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --output results/run.json
```

To run the unit and verification test suite (including synthetic timestamp checks for multi-token, single-token, and zero-token cases, exact prompt-length assertions, and JSON result validation):

```sh
pytest
```

## Workloads

Workloads are configured in `workloads/workloads.json` and built deterministically from `workloads/prompt.txt`:

| Workload | Target prompt tokens | Maximum output tokens (cap) |
|---|---:|---:|
| `short` | 128 | 64 |
| `medium` | 512 | 128 |
| `long` | 2048 | 256 |

Prompts are tokenized directly with the model's tokenizer without a chat template. If the tokenizer defines `bos_token_id`, it is placed once at index 0, and token IDs from `workloads/prompt.txt` are repeated and truncated so the total input token sequence passed to the model equals `128`, `512`, or `2048` exactly (verified via runtime `assert` before generation).

## Measurement boundaries & metric definitions

All timing uses `time.perf_counter()`. Before each request, pending Metal GPU work is completed (`mx.synchronize()`), the MLX cache is cleared (`mx.clear_cache()`), and the peak memory counter is reset (`mx.reset_peak_memory()`).

Let:
- `t0`: timestamp immediately before starting the streaming token generator (`generate_step` with `prompt_cache=None` and greedy `argmax` sampling)
- `t1`: timestamp when the first generated token is materialized on host (`int(token_id)`)
- `tN`: timestamp when the last generated token (`N`-th token) is materialized
- `t_end`: timestamp after the generation loop exits and `mx.synchronize()` completes all pending device work
- `N`: actual number of generated tokens (if an EOS token in `tokenizer.eos_token_ids` is encountered, generation stops immediately and the EOS token is not counted in `N`)

| Output field | Unit | Definition & boundaries |
|---|---|---|
| `ttft_ms` | milliseconds | `(t1 - t0) * 1000`. Includes prompt prefill and generating the first output token (`null` if `N == 0`). |
| `total_latency_s` | seconds | `t_end - t0`. End-to-end request generation time. **Excludes** model download, weight loading, prompt tokenization, and writing JSON results. |
| `decode_tokens_per_sec` | tokens/second | `(N - 1) / (tN - t1)`. Steady-state decode rate after the first token (`null` if `N < 2` or `tN <= t1`). |
| `prompt_tokens` | tokens | Exact count of input token IDs passed to the model (including the leading BOS token). |
| `output_tokens` | tokens | Actual generated token count `N` (`N <= max_output_tokens`). |
| `peak_memory_gib` | GiB (`bytes / 2^30`) | **Peak MLX allocated memory** (`mx.get_peak_memory() / 2^30`) after resetting the peak counter prior to the run. Includes resident model weights and KV cache allocations inside MLX's Metal allocator; it is **not** total macOS system memory or process RSS. |
| `stop_reason` / `early_stop` | string / boolean | `"length"` (`early_stop: false`) when `output_tokens == max_output_tokens`, or `"eos"` (`early_stop: true`) if stopped early on an EOS token. |
