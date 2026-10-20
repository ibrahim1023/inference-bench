# InferenceBench — Build Spec

## Goal

Build a small command-line tool that benchmarks one language model locally on a MacBook Pro M4 Pro. The owner has finished Chapter 1 of learn-inference.com. Prioritize readable code and trustworthy measurements over features.

**Deliver the working repository and a short report using real measurements from the owner's Mac.** If that hardware is unavailable, deliver the tool, explain how to run it, and leave measurements pending. Never invent results.

## Scope and requirements

- Use Python 3.11+ and **one backend only**: MLX through `mlx-lm` by default. llama.cpp is an acceptable substitute if MLX cannot be used; document the reason and equivalent measurement method. Do not implement both.
- Use one small, publicly downloadable, 4-bit model, around 3B parameters, that fits available memory. Make its model identifier/path configurable. Record the exact model, revision, and quantization.
- Run directly on macOS with Apple Silicon acceleration. Downloading dependencies and model weights is allowed; benchmark inference must run locally.
- Use streaming token generation, a monotonic high-resolution timer, and greedy decoding. Run one request at a time with no shared prompt/KV cache between requests.
- Keep dependencies minimal and record the versions actually tested in `requirements.txt`.

## Workloads

Store these settings in one JSON file:

| Name | Prompt tokens | Maximum output tokens |
|---|---:|---:|
| short | 128 | 64 |
| medium | 512 | 128 |
| long | 2048 | 256 |

Create deterministic prompts from a fixed text fixture. Tokenize using the selected model's tokenizer, repeat/truncate token IDs to the target length, and pass those IDs directly to generation. Count every token actually passed to the model, including any required special tokens. Avoid a chat template for this synthetic benchmark. Assert the final prompt length matches the target.

Output lengths are **caps**: respect normal end-of-sequence behavior, record actual output tokens and stop reason, and clearly flag early stops. Do not pad output or pretend the cap was reached.

## Measurements

Prepare/tokenize prompts and load the model before timing. Complete any pending backend work before starting each timer. Use token events rather than printed words or non-empty text chunks; handle asynchronous/lazy execution so timestamps reflect completed token generation.

Let `t0` be immediately before generation starts, `t1` the first generated token, `tN` the last generated token, and `t_end` completion of the generation iterator after pending device work completes. Let `N` be the actual number of generated tokens, counted consistently with the backend.

| Metric | Definition |
|---|---|
| TTFT, milliseconds | `(t1 - t0) × 1000`; includes prefill and first-token generation |
| Total latency, seconds | `t_end - t0`; excludes model loading, downloads, tokenization, and writing results |
| Decode tokens/sec | `(N - 1) / (tN - t1)`; excludes the first token; `null` if fewer than two tokens or zero elapsed time |
| Prompt tokens | Actual input token count |
| Output tokens | Actual generated token count; document treatment of EOS |
| Peak memory, GiB | For MLX, reset its peak counter before each request and read peak MLX allocated memory after completion; report bytes divided by `2^30` |

Label memory as **peak MLX allocated memory**, including resident model allocations; it is not total Mac memory or total process memory. If using llama.cpp, document and label the alternative memory source and its limits. Fail clearly if a required metric cannot be measured; do not silently substitute an unrelated metric.

## Repository

```text
inference-bench/
├── README.md
├── benchmark.py
├── requirements.txt
├── workloads/
│   ├── workloads.json
│   └── prompt.txt
├── results/
│   └── .gitkeep
└── report.md
```

Keep the implementation in one readable script with small functions. No framework or backend abstraction is needed.

## Implementation steps

1. Set up the environment and load the selected model. Confirm one short generation works locally.
2. Load workload settings and build exact-length token inputs. Validate counts before generation.
3. Implement streaming measurements and memory collection using the definitions above.
4. For each workload, run one unmeasured warm-up followed by three measured runs. Keep model weights loaded, but start each request with fresh prompt/KV state. Reset memory measurement for every run.
5. Provide this interface, or its clearly documented equivalent:
   ```sh
   python benchmark.py --model <model-id-or-path> --output results/run.json
   ```
6. Print a compact table with the median TTFT, total latency, decode rate, and peak memory for each workload. Save every individual run and run metadata to JSON. Report early stops alongside actual output counts.
7. Run all three workloads on the target Mac, inspect the results, and write the README and report.

JSON must include workload name, repetition, target and actual token counts, all six metrics, stop reason, and units. Record timestamp, model/revision, backend/package versions, macOS version, chip, available RAM, generation settings, and warm-up/repetition counts. Errors should identify the workload and exit with a nonzero status.

## Acceptance criteria

- A new user can set up and run the tool by following the README.
- One command executes all three workloads and saves nine measured runs, excluding warm-ups.
- Prompt counts equal 128, 512, and 2048; output counts never exceed their caps.
- Timings follow the definitions above; first-token decode division and zero/one-token cases are handled correctly.
- Console medians agree with the saved raw results. Peak memory has a clear source, scope, and unit.
- A small focused check verifies timing calculations using synthetic timestamps, including zero/one-token cases. Prompt-length assertions run for every workload.
- Results come from actual hardware execution. If execution is unavailable, the README/report explicitly says validation is pending.

## README and report

**README:** explain the purpose, prerequisites, environment setup, model download, exact run command, output fields, and measurement boundaries. Mention that initial downloads/load time are excluded. Suggest plugging in the Mac and closing heavy applications for comparable runs.

**Report:** include the machine/model configuration, one table of measured medians, and 3–5 observations. Discuss how prompt length relates to TTFT, whether decode rate changes, and memory behavior. Separate observations from hypotheses. Because both prompt and output lengths vary across these workloads, do not attribute total-latency differences to prompt length alone. No speed targets or claims that one backend is faster.

## Non-goals

No Docker, Kubernetes, Grafana, cloud GPUs, distributed serving, UI, API server, concurrent requests, batching, optimization work, backend comparisons, or model-quality evaluation. Finish this baseline before proposing extensions.

## Backend references

Use the installed version's supported APIs; check signatures rather than copying outdated examples:

- [MLX-LM documentation](https://github.com/ml-explore/mlx-lm)
- [MLX-LM benchmark reference](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py)
