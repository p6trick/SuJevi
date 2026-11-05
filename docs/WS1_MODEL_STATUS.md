# WS1 model status

Last updated: 2026-11-05 (Asia/Seoul)

Status values are `Pending`, `Code Ready`, `Awaiting User Execution`,
`Success`, `Failed`, and `Blocked`. A step is marked `Success` only from a
user-provided execution result.

| Order | Model | Research | Environment | Download | Native inference | Noul/Choice/Score |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kev-4B | Success | Success | Success | Success | Success |
| 2 | Jeff | Pending | Pending | Pending | Pending | Pending |
| 3 | Decider | Pending | Pending | Pending | Pending | Pending |
| 4 | Laya | Pending | Pending | Pending | Pending | Pending |
| 5 | Bosun | Pending | Pending | Pending | Pending | Pending |

## Kev-4B research record

- Official code: `jaredpalmer/kev`, release `kev-1.0`, pinned to
  `6b719c3c3f367295f6ef336f4f751cf5ff970abc`.
- Checkpoint: `jaredpalmer/kev-4b@v1.0`, currently resolving to
  `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`. The model card identifies
  `139fdd94` as the released weights revision.
- Base: `Qwen/Qwen3.5-4B-Base`, pinned to
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Runtime: official Kev source with its committed `uv.lock`; local Python uses
  `Checkpoint`, `LoadOptions`, `DecisionModel`, and the official API schema
  helpers directly. An HTTP server is not required.
- Native schema: TypeSafe-compatible System One request and response objects.
- Native question types: `noul`, `choice`, and `score`.
- Calibration: checkpoint temperature is applied by default. Native option
  distributions must be retained without conversion during WS1 validation.
- License: Apache-2.0 for Kev and the Qwen base.
- Hardware target: NVIDIA H200. No environment, CUDA, weight, or inference
  command has been executed by Codex.

### User-verified environment

- Verified on 2026-11-05 from the user-provided environment report.
- Python 3.12.16 and all declared package constraints passed.
- PyTorch 2.8.0 sees CUDA 12.8, one NVIDIA H200, compute capability 9.0,
  and 139.8 GiB device memory.
- `kev` is importable and the environment checker reported `ok: true` with
  no failures.
- At this point in the record, model loading and inference had not yet been
  executed. See the successful validation below.

### User-verified download

- Verified on 2026-11-05 from the user-provided `hf download` output.
- Kev-4B resolved to the pinned snapshot
  `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`.
- Qwen3.5-4B-Base resolved to the pinned snapshot
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- The project-local Hugging Face cache occupies 8.9 GiB as reported by the
  user.
- At this point in the record, native inference had not yet been executed. See
  the successful validation below.

### Local inference attempt history

- The first user execution on 2026-11-05 stopped before model loading because
  `sujevi.__init__` eagerly imported OneJev and its unavailable `qev`
  dependency. This was a cross-model packaging issue, not a Kev or CUDA
  failure.
- The package exports are now lazy, so importing `sujevi.kev_local` does not
  import OneJev or Jev-Omni dependencies. The subsequent retry succeeded.

### User-verified local Python inference

- Verified on 2026-11-05 from the user-provided artifacts and execution log.
- Loaded locally in BF16 on NVIDIA H200 with the adapter unmerged and no
  quantization.
- Exposed components:
  `peft.peft_model.PeftModelForFeatureExtraction`,
  `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel`, and
  `kev.model.PointerHead`.
- Applied the checkpoint calibration temperature `2.406050072164233`.
- Noul distribution: false `0.0663387626`, true `0.9336612225`.
- Choice distribution: billing `0.7548326850`, shipping `0.1123690531`,
  returns `0.1327982396`; selected billing.
- Score distribution: level 0 `0.0368090011`, level 1 `0.1693245769`, level 2
  `0.7938665152`; expected score `1.7571`.
- Each full-precision distribution sums to 1 within floating-point error.
- The single smoke request used 127 input tokens and reported 649.5 ms. This is
  a correctness check, not a performance benchmark.
- `causal_conv1d` and `flash-linear-attention` were absent, so Transformers used
  its correct but slower reference PyTorch kernels. This does not invalidate
  the probabilities; optimized-kernel installation can be evaluated separately
  before throughput or fine-tuning benchmarks.

## Current gate

Kev-4B WS1 validation is complete. Preserve the code, request, native response,
full distributions, and status record in Git before starting Jeff research.
