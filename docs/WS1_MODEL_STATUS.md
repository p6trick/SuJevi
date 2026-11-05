# WS1 model status

Last updated: 2026-11-05 (Asia/Seoul)

Status values are `Pending`, `Code Ready`, `Awaiting User Execution`,
`Success`, `Failed`, and `Blocked`. A step is marked `Success` only from a
user-provided execution result.

| Order | Model | Research | Environment | Download | Native inference | Noul/Choice/Score |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kev-4B | Success | Success | Success | Success | Success |
| 2 | Jeff-1.1-Qwen-4B | Success | Success | Success | Success | Success |
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

## Jeff-1.1-Qwen-4B research record

- Official code: `j-geuter/Jeff`, release `v1.3.0`, pinned to
  `2c446ca3249f31c83cc4a230f3aed605858da330`.
- Checkpoint: `jgeuter/Jeff-1.1-Qwen-4B`, pinned to
  `5401940b5a8c959cba82187ebfa2dce5a21abc0a`.
- Base: `Qwen/Qwen3.5-4B`, pinned to
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Paper: *Releasing the Inner Jev: A Simple Recipe for Open Decision Models*,
  arXiv:2610.17205v1. Jeff 1.1 follows its LoRA and residual decision-head
  method but adds data, longer training, model souping, and bidirectional
  attention; the current paper version directly reports the 1.0 suite.
- Runtime: official `jeff.JeffModel` used directly in local Python. HTTP is not
  required. BF16 weights are retained without quantization, reasoning is off,
  and Noul commitment is disabled so calibrated probabilities are preserved.
- Representation: raw logits, raw probabilities, calibrated probabilities,
  and ordered option-ID mappings are retained. Calibrated probabilities are
  the default representation.
- Fine-tuning: the repository has no official training script. WS1 records
  this limitation and does not implement fine-tuning.
- License: checkpoint CC BY-NC 4.0; code and Qwen base Apache-2.0.
- Environment, checkpoint download, model loading, and inference have not been
  executed by Codex. They remain user-execution gates.

### User-verified Jeff environment

- Verified on 2026-11-05 from the user-provided environment report.
- Python 3.12.16 and all exact official package pins passed.
- Official `jeff` 1.3.0 is importable.
- PyTorch 2.13.0+cu130 sees CUDA 13.0, one NVIDIA H200, compute capability
  9.0, and 139.8 GiB device memory.
- The environment checker reported `ok: true` with no failures.
- At this point in the record, checkpoint download, model loading, and
  inference had not yet been executed. See the successful download below.

### User-verified Jeff download

- Verified on 2026-11-05 from the user-provided download path, directory
  listing, and checksum log.
- The checkpoint resolved to the pinned snapshot
  `5401940b5a8c959cba82187ebfa2dce5a21abc0a`.
- Every file listed by the release's `SHA256SUMS`, including both BF16 model
  shards, the separate adapter, decision head, tokenizer, Jeff configuration,
  and serving rule, passed `sha256sum --check`.
- The project-local Hugging Face cache occupies 18 GiB as reported by the
  user. This includes the previously downloaded model artifacts.
- Model loading and inference have not yet been executed.

### Jeff Noul attempt history

- The first user-executed Noul attempt on 2026-11-05 loaded all 426 weight
  entries successfully, then stopped during the first forward pass before any
  logits or probabilities were produced.
- Triton reported `Failed to find C compiler` while building its CUDA runtime
  helper. This is an environment prerequisite documented by the official Jeff
  repository, not a checkpoint, CUDA-driver, GPU-memory, or model-loading
  failure.
- The environment checker originally validated only Python packages and CUDA,
  so it incorrectly allowed the environment gate to pass without a compiler.
  It now checks for `CC`, `cc`, `gcc`, or the Conda compiler executable.
- No Noul output artifact was produced. Install and validate the compiler
  before retrying inference.

### User-verified compiler remediation

- Verified on 2026-11-05 from the user-provided environment report after
  installing the Conda C compiler.
- `CC` was not exported automatically, but the environment exposes
  `/opt/conda/envs/jeff/bin/cc` on `PATH`; the environment checker accepts a
  standard `cc` executable and reported `c_compiler.available: true`.
- All pinned Python packages and CUDA checks remained unchanged and valid.
- The full environment report returned `ok: true` with no failures. The Noul
  smoke test may now be retried with `CC` set explicitly to this executable.

### User-verified Jeff Noul inference

- Verified on 2026-11-06 from the user-provided native response,
  representation artifact, and execution log.
- Loaded the pinned local checkpoint in BF16 on NVIDIA H200, without
  quantization or reasoning and with Noul commitment disabled.
- Exposed components:
  `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM`,
  `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel`, and
  `jeff.head.DecisionHead`.
- Option order was `true`, `false`. Raw logits were `20.8772201538` and
  `17.3135967255` respectively.
- Raw distribution: true `0.9724448919`, false `0.0275551639`; its sum differs
  from 1 by about `5.6e-8`, within floating-point conversion error.
- Calibrated distribution: true `0.9478729023`, false `0.0521270977`; it sums
  to 1, and calibrated true exactly matches the native `noul` value.
- The stored default model representation is the calibrated distribution, and
  the ordered option-ID mapping preserves the same values.
- The request used 120 input tokens and reported 1721.9 ms. This first valid
  request includes runtime warm-up effects and is not a performance benchmark.
- `causal_conv1d` and `flash-linear-attention` were absent, so Transformers
  used its correct but slower reference PyTorch implementations. This does not
  invalidate the outputs.

### User-verified Jeff Choice inference

- Verified on 2026-11-06 from the user-provided native response,
  representation artifact, and execution log.
- Option order was `billing`, `shipping`, `returns`. Raw logits were
  `22.1944580078`, `13.6211662292`, and `13.4856719971` respectively.
- Raw distribution: billing `0.9996459484`, shipping `0.0001890222`, returns
  `0.0001650701`; its sum differs from 1 by about `4.1e-8`, within
  floating-point conversion error.
- Calibrated distribution: billing `0.9975366009`, shipping `0.0012963235`,
  returns `0.0011670756`; it sums to 1.
- The native choice and calibrated argmax are both `billing`. The ordered
  option-ID mapping and default calibrated representation preserve the same
  values.
- The request used 149 input tokens and reported 844.8 ms. This is a
  correctness check, not a performance benchmark.
- The same correct optional-kernel fallback paths were used.

### User-verified Jeff Score inference

- Verified on 2026-11-06 from the user-provided native response,
  representation artifact, and execution log.
- Score-level order was `0`, `1`, `2`. Raw logits were `14.0873641968`,
  `15.8100881577`, and `20.2683811188` respectively.
- Raw distribution: level 0 `0.0020404703`, level 1 `0.0114261461`, level 2
  `0.9865334034`; its sum differs from 1 by about `2.0e-8`, within
  floating-point conversion error.
- Calibrated distribution: level 0 `0.0049453377`, level 1 `0.0215575836`,
  level 2 `0.9734970787`; it sums to 1.
- The native score and calibrated argmax are both level `2`. The ordered
  option-ID mapping and default calibrated representation preserve the same
  values.
- The request used 137 input tokens and reported 832.7 ms. This is a
  correctness check, not a performance benchmark.
- The same correct optional-kernel fallback paths were used.

## Current gate

Kev-4B and Jeff-1.1-Qwen-4B WS1 validation are complete. Preserve the Jeff
environment profile, pinned source metadata, runtime, three smoke requests,
six output artifacts, and status record in Git before starting Decider.
