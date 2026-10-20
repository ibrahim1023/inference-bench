# InferenceBench — Benchmark Report

## 1. Machine, Backend, and Model Configuration

All measurements in this report were captured from actual local execution on the target Mac (`results/run.json`, timestamp `2026-10-19T13:12:03Z`).

| Category | Setting / Value |
|---|---|
| **Chip** | Apple M4 Pro |
| **System Memory** | 24.0 GiB unified memory (8.27 GiB available at benchmark start) |
| **OS** | macOS 27.0.1 (Build `26A434`) |
| **Python & Backend** | Python `3.12.11`, `mlx` `0.32.4`, `mlx-lm` `0.32.2` (`Device(gpu, 0)`) |
| **Model** | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| **Model Revision** | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` |
| **Quantization** | 4-bit (`group_size: 64`, `bits: 4`, `model_type: llama`) |
| **Decoding & Cache Policy** | Greedy (`argmax`), streaming (`generate_step`), batch size 1, fresh KV cache per request (`prompt_cache=None`), no chat template |
| **Protocol** | 1 unmeasured warm-up run followed by 3 measured runs per workload |
| **EOS Handling** | Stop immediately if token ID is in `tokenizer.eos_token_ids`; EOS token is excluded from `output_tokens` |

---

## 2. Measured Medians Across Workloads

Each cell reports the median across the 3 measured runs saved in `results/run.json`. Every run reached its output token cap (`0/3` early stops across all three workloads).

| Workload | Prompt Tokens | Output Tokens (runs) | Early Stops | Median TTFT (ms) | Median Total Latency (s) | Median Decode Rate (tokens/sec) | Median Peak MLX Allocated Memory (GiB)* |
|---|---:|---|---:|---:|---:|---:|---:|
| `short` | 128 | 64, 64, 64 (cap 64) | 0/3 | 156.68 | 0.703 | 117.08 | 2.061 |
| `medium` | 512 | 128, 128, 128 (cap 128) | 0/3 | 573.79 | 1.701 | 113.30 | 2.322 |
| `long` | 2048 | 256, 256, 256 (cap 256) | 0/3 | 2352.46 | 4.850 | 102.47 | 2.617 |

*\*Memory is **peak MLX allocated memory** (`mlx.core.get_peak_memory() / 2^30`), reset before each request. It includes resident 4-bit model weights plus per-request KV cache and activation buffers allocated by MLX on Metal—not total macOS memory or process RSS.*

### Raw Run Consistency

Run-to-run variance across the 3 measured repetitions per workload was low:
- **`short`:** TTFT `156.28–157.22 ms`, total latency `0.701–0.706 s`, decode rate `116.57–117.38 tok/s`, peak memory `2.061 GiB`.
- **`medium`:** TTFT `570.86–574.38 ms`, total latency `1.697–1.715 s`, decode rate `112.25–113.92 tok/s`, peak memory `2.322 GiB`.
- **`long`:** TTFT `2341.59–2355.94 ms`, total latency `4.844–4.854 s`, decode rate `101.87–102.86 tok/s`, peak memory `2.617 GiB`.

---

## 3. Observations vs. Hypotheses

### Observation 1: TTFT scales nearly linearly (With Slight Super-Linearity at 2048 Tokens) With Prompt Length
- **Observation:** Increasing prompt length by `4x` from `short` (`128` tokens) to `medium` (`512` tokens) increases median TTFT by `3.66x` (`156.68 ms` to `573.79 ms`, or `~1.22 ms/token` to `~1.12 ms/token`). Increasing prompt length by another `4x` from `medium` (`512` tokens) to `long` (`2048` tokens) increases median TTFT by `4.10x` (`573.79 ms` to `2352.46 ms`, or `~1.15 ms/token`). Across the full `16x` prompt increase (`128` to `2048`), TTFT increases by `15.01x`.
- **Hypothesis:** During prefill, transformer linear projections scale as $O(L)$ with prompt length $L$, while self-attention scales as $O(L^2)$. At `128` tokens, fixed per-launch overhead slightly inflates the per-token cost (`1.22 ms/token` vs. `1.12 ms/token` at `512` tokens). Between `512` and `2048` tokens, the GPU is solidly compute-bound on matrix multiplications and quadratic attention cost begins to add a small surcharge (`4.10x` time for `4.00x` tokens).

### Observation 2: Steady-State Decode Rate Declines Gradually as Total Context Length Grows
- **Observation:** Median decode throughput decreases by `3.2%` from `short` (`117.08 tok/s`, sequence range `128–192` tokens) to `medium` (`113.30 tok/s`, sequence range `512–640` tokens), and by an additional `9.6%` from `medium` to `long` (`102.47 tok/s`, sequence range `2048–2304` tokens)—an overall `12.5%` reduction from `short` to `long`.
- **Hypothesis:** Single-stream greedy decoding (`batch_size=1`) is memory-bandwidth bound: every generated token must read the ~1.8 GiB of 4-bit model weights plus the KV cache for all prior tokens. As the active context grows from `~160` tokens in `short` to `~2176` tokens in `long`, reading and attending over the larger KV cache at each step consumes additional unified memory bandwidth and attention compute, gradually lengthening inter-token latency from `8.54 ms/token` (`short`) to `9.76 ms/token` (`long`).

### Observation 3: Total Latency Reflects Both Longer Prefill and Higher Output Token Counts
- **Observation:** Median total latency rises from `0.703 s` (`short`: `128` in / `64` out) to `1.701 s` (`medium`: `512` in / `128` out) and `4.850 s` (`long`: `2048` in / `256` out). Breaking total latency into TTFT vs. decode time (`total_latency_s - ttft_s`):
  - `short`: `0.157 s` TTFT (`22.3%`) + `0.547 s` for 63 decode tokens (`77.7%`).
  - `medium`: `0.574 s` TTFT (`33.7%`) + `1.127 s` for 127 decode tokens (`66.3%`).
  - `long`: `2.352 s` TTFT (`48.5%`) + `2.498 s` for 255 decode tokens (`51.5%`).
- **Hypothesis:** Because both `prompt_tokens` (`4x` per tier) and `max_output_tokens` (`2x` per tier) increase simultaneously across the three workloads, total latency growth (`2.42x` from `short` to `medium`, `2.85x` from `medium` to `long`) cannot be attributed to prompt length alone. Prefill grows roughly `4x` per tier while decode time slightly more than doubles per tier (`2x` more output tokens compounded by the lower decode tok/s at longer contexts), shifting TTFT's share of total latency from `22.3%` in `short` to `48.5%` in `long`.

### Observation 4: Peak MLX Allocated Memory Grows Sub-Linearly Because Resident Weights Dominate
- **Observation:** Peak MLX allocated memory is identical across repetitions within each workload and increases from `2.061 GiB` (`short`, 192 total tokens) to `2.322 GiB` (`medium`, 640 total tokens, `+261 MiB`) and `2.617 GiB` (`long`, 2304 total tokens, `+295 MiB` over `medium`, `+556 MiB` over `short`).
- **Hypothesis:** The ~1.8 GiB 4-bit Llama-3.2-3B weights remain resident in MLX's Metal allocator across all runs and form a fixed baseline. The incremental `+0.556 GiB` between `short` and `long` comes from the larger prefill activation tensors (scaling with prompt chunk size up to `2048` tokens) and the unquantized FP16/BF16 KV cache growing across 28 transformer layers over `2304` total sequence positions.
