# WS1 model status

Last updated: 2026-11-07 (Asia/Seoul)

Status values are `Pending`, `Code Ready`, `Awaiting User Execution`,
`Success`, `Failed`, `Blocked`, and `Excluded`. A step is marked `Success`
only from a user-provided execution result. `Excluded` records a deliberate
model-pool decision and requires no environment or runtime execution.

| Order | Model | Research | Environment | Download | Native inference | Noul/Choice/Score |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kev-4B | Success | Success | Success | Success | Success |
| 2 | Jeff-1.1-Qwen-4B | Success | Success | Success | Success | Success |
| 3 | Decider | Excluded | Excluded | Excluded | Excluded | Excluded |
| 4 | Laya | Success | Success | Success | Success | Success |
| 5 | Bosun | Success | Success | Success | Success | Success |
| 6 | Jeva-4B | Success | Success | Success | Success | Success |

## Jeva-4B research and environment record

- Paper: *Jeva: System One Models from First Principles*, arXiv:2610.18773v2.
- Official code: `LIQIIIII/Jeva`, revision
  `e2104e0720100f979f2cdd456a1cbbee96da0abd`.
- Checkpoint: `LIQIIIII/Jeva-4B`, revision
  `25ca454b3522615a243c06703ba410ecf152a128`.
- Official Python runtime supports local Noul, Choice, and Score; an HTTP
  server is not required. The official training CLI supports head and LoRA
  modes. Code and checkpoint are Apache-2.0.
- User reported Python 3.12.16, PyTorch 2.9.1+cu128, NVCC 12.8.93,
  NVIDIA H200 with BF16, and successful official forward-kernel doctor checks.
- `pip check` passed with `PYTHONPATH` unset. With SuJevi's source path on
  `PYTHONPATH`, unrelated SuJevi package requirements were reported missing.
- The user executed the SuJevi Jeva environment profile: all package
  constraints and the CUDA 12.8 runtime check passed with `ok: true`.
  Kernel execution was independently verified with the official doctor.
- The user reported successful download of the pinned checkpoint, including
  backbone shards, adapter, tokenizer, config and decision head. Full cache
  size was reported as 32 GiB. No checksum comparison was reported.
- The user verified local BF16 Noul inference with the unmerged PEFT adapter.
  Raw scores were false -1.0140837431, true -0.7854450345; their softmax was
  false 0.4430879951, true 0.5569119453. Learned decision temperature was
  0.1104928032; the resulting native distribution was false 0.1121203825,
  true 0.8878796101, matching the native Noul and option mapping. Separate
  post-hoc calibration has not been established.
- The user verified a three-option Choice request. The native distribution
  was billing 0.9608018398, shipping 0.0045975656, and returns 0.0346005298;
  the selected option, captured logits and option-ID mapping agree. No input
  tokens were truncated.
- The user verified Score levels 0/1/2 with native probabilities 0.0782801658,
  0.2710691690 and 0.6506507397. The reported expected score 1.5723706484
  equals the probability-weighted level values. The option mappings match
  the native output, input was not truncated, and the adapter was unmerged.
- The first Noul request reported 56.2 seconds; this is a cold request and
  does not establish steady-state performance. The stored head null
  probability is an internal coverage output, not the native Noul null field.

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

## Decider exclusion record

Decider was excluded from the Surrogate Model Pool by user decision on
2026-11-06. No Decider environment, checkpoint, local runtime, smoke test, or
output artifact is required for WS1.

The research review found no accompanying paper or official citation entry.
The latest `Mapika/decider-4b` v3 release also uses a deliberately low default
Noul temperature (`T=0.05`) to avoid JevBench abstention, while its model card
states that the resulting yes/no probabilities are not calibrated. Fitted
temperatures are published for all three question types, so a normalized
wrapper would be possible, but it would not represent the release default.
The repository includes supervised training and data-building code, although
the exact current v3 rank-64 LoRA stage and complete mixture-v2 recipe are not
fully reproducible from a single public training entry point.

## Laya research and environment record

- Official code: `NandhaKishorM/laya`, release `v0.4.24`, pinned to
  `9d3fcc8924ef47b71510f77e19659f97ade0eec4`.
- Checkpoint: `convaiinnovations/laya`, pinned to
  `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- Architecture: ModernBERT-large plus a two-layer typed-decision head, about
  421M parameters, with native Noul, Choice, and Score outputs.
- Runtime: official local `laya.load` / `laya.Agent`; no HTTP service,
  quantization, ONNX conversion, or TileLang path is required.
- Fine-tuning: official `laya-train` CLI and `laya.train.finetune` Python API
  support RLCD, soft cross-entropy, full-encoder, and head-only training.
- Paper: no first-party Laya methods paper was identified as of 2026-11-06.
- License: Apache-2.0 for both code and checkpoint.

### User-verified environment

- Verified on 2026-11-06 from the user-provided environment report.
- Python 3.12.16 in `/opt/conda/envs/laya/bin/python`.
- PyTorch 2.13.0+cu130 with CUDA runtime 13.0.
- Transformers 5.17.0, Laya 0.4.24, and VegaML 0.15.0.
- CUDA available on one NVIDIA H200, compute capability 9.0, with 139.8 GiB.
- All declared package constraints passed; no failures were reported.

### User-verified checkpoint download

- Verified on 2026-11-06 from the user-provided download and file report.
- Snapshot resolved to the pinned revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- All five files required by the official Agent allow-list were present.
- `model.safetensors` size: 842,609,210 bytes.
- `model.safetensors` SHA-256:
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- The shared project Hugging Face cache was 19 GiB after download.

### User-verified Noul smoke test

- Verified on 2026-11-06 using the official eager Agent path on the H200.
- Backbone: `ModernBertModel`; decision head: `TransformerEncoder`; option
  scorer: `Sequential`.
- Raw logits `[false, true]`: `[-2.125, 1.4453125]`.
- Raw distribution `[false, true]`:
  `[0.027376487851142883, 0.9726234674453735]`.
- Official `noul:2` temperature: `1.983399510383606`.
- Temperature-scaled distribution `[false, true]`:
  `[0.14183920621871948, 0.8581608533859253]`.
- Native Noul `0.8582` equals the full-precision P(true) rounded to four
  decimals. Neither state nor option content was truncated.
- The expected `choice:11+` clamp warning appeared and does not affect this
  Noul bucket.
- `action.act_probability` was `1.0`; it is retained as native output but is
  not used as a model representation or routing signal because the official
  model card marks the action head as not usable.

### User-verified Choice smoke test

- Verified on 2026-11-06 using the same eager BF16 runtime.
- Ordered option IDs: `[billing, shipping, returns]`; no state or option
  content was truncated.
- Raw logits: `[2.4375, -2.15625, 2.375]`.
- Raw distribution:
  `[0.5129446983337402, 0.005188361741602421, 0.4818669259548187]`.
- Official `choice:3-5` temperature: `1.7601518630981445`.
- Temperature-scaled distribution:
  `[0.49051839113235474, 0.03607510030269623, 0.47340652346611023]`.
- Native probabilities `[0.4905, 0.0361, 0.4734]` match the full-precision
  distribution rounded to four decimals, and native choice `billing` matches
  its argmax.
- Native `confidence=0.2506` is normalized-entropy confidence, whereas
  `answer_confidence=0.4905` is calibrated top probability. WS1 uses neither
  scalar in place of the complete calibrated distribution.

### User-verified Score smoke test

- Verified on 2026-11-06 using the same eager BF16 runtime.
- Ordered level IDs: `[0, 1, 2]`; no state or option content was truncated.
- Raw logits: `[-2.421875, -2.28125, 3.890625]`.
- Raw distribution:
  `[0.0018064472824335098, 0.0020792081486433744, 0.9961143732070923]`.
- Official `score:3-5` temperature: `1.2514300346374512`.
- Temperature-scaled distribution:
  `[0.006359559949487448, 0.0071158926002681255, 0.9865245819091797]`.
- Native probabilities `[0.0064, 0.0071, 0.9865]` match the full-precision
  distribution rounded to four decimals.
- The full-precision expected score is `1.9801650564186275`; native score
  `1.9802` matches it rounded to four decimals.

## Bosun v3.1 1.7B research and environment record

- Checkpoint: `Hanno-Labs/bosun-v3.1-1.7b`, pinned to
  `1d8dc82a20e4a32ed60927a47272d6efff48eed2`.
- Base: `Qwen/Qwen3-1.7B`, pinned to
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Architecture: ordinary Qwen3 causal base, rank-16 LoRA, and 256 learned
  decision-token rows; up to 255 candidates plus one null slot.
- Runtime: checkpoint-provided `BosunForDecision` loaded through local
  Transformers Python with reviewed remote code. `decision_logits()` exposes
  raw slot logits, and `predict()` supports Noul, Choice, and Score without an
  HTTP service.
- Probability rule: masked softmax over only the presented decision slots,
  followed by mapping back to caller candidate order.
- Calibration boundary: the release publishes no post-hoc temperature. The
  common-schema calibrated distribution is therefore identical to the
  official masked-softmax distribution until routing-domain calibration is
  fitted and validated.
- Fine-tuning: the release records the LoRA and distribution-target recipe but
  provides neither an official training entry point nor the full 130,000-row
  training corpus. Reimplementation is technically possible; exact public
  reproduction is not.
- Paper: no first-party Bosun methods paper was identified as of 2026-11-06.
  The official release article and model card are the primary method sources.
- License: Apache-2.0 for the checkpoint remote code and Qwen base.
- Environment code is ready for user execution. No environment installation,
  model download, model loading, or GPU inference was executed by Codex.

### Environment attempt history

- The first user-executed environment attempt on 2026-11-06 successfully
  created Python 3.12.16 and installed PyTorch 2.13.0+cu130.
- PyTorch detected CUDA 13.0 and the NVIDIA H200 correctly, but the remaining
  installation was rejected before changing the environment: Transformers
  4.57.0 requires `huggingface-hub>=0.34.0,<1.0`, while the initial profile
  incorrectly pinned Hub 1.32.0.
- The profile now pins `huggingface_hub==0.36.0` and explicitly installs NumPy
  2.5.3. The existing `bosun` environment can be repaired in place; recreation
  and PyTorch reinstallation are not required.

### User-verified environment

- Verified on 2026-11-06 from the user-provided retry report.
- Python 3.12.16 in `/opt/conda/envs/bosun/bin/python`.
- PyTorch 2.13.0+cu130 detected CUDA 13.0 and one NVIDIA H200 with compute
  capability 9.0 and 139.8 GiB device memory.
- Transformers 4.57.0, PEFT 0.21.0, Safetensors 0.8.0, Hugging Face Hub
  0.36.0, Accelerate 1.15.0, and NumPy 2.5.3 all matched the profile. The
  environment checker returned `ok: true` with no failures.
- The `pip check` messages about `sujevi 0.1.0` came from the repository's
  `src/sujevi.egg-info`, exposed by an already exported `PYTHONPATH`. They do
  not indicate that the Bosun environment installed the monolithic SuJevi
  dependency set. A clean check must temporarily remove `PYTHONPATH`.

### User-verified checkpoint download

- Verified on 2026-11-06 from the user-provided downloader output and file
  listings.
- Bosun resolved to the pinned snapshot
  `1d8dc82a20e4a32ed60927a47272d6efff48eed2`; Qwen3-1.7B resolved to
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- The Bosun adapter (`69,782,384` bytes), learned decision embeddings
  (`2,097,336` bytes), tokenizer, model/configuration Python files,
  `serving.json`, and content-addressed `manifest.json` were all present.
- The Qwen base contained its index and two Safetensors shards of
  `3,441,185,608` and `622,329,984` bytes.
- The shared project Hugging Face cache was 23 GiB. The small per-snapshot
  `du` values are expected because Hub snapshots contain symlinks into the
  shared blob store.

### Local runtime code

- `BosunLocalRuntime` validates both pinned revisions and verifies every file
  hash declared by the official Bosun manifest before remote-code import.
- The official `BosunForDecision` is loaded locally in BF16 with the original
  Qwen weights, unmerged PEFT adapter, learned input/output decision rows,
  `local_files_only=True`, and no HTTP server or quantization.
- The wrapper calls official `predict()` and preserves raw caller-aligned
  logits, masked-softmax probabilities, the identity calibrated field,
  candidate-to-slot mapping, slot-order logits/probabilities, decision-token
  IDs, and ordered option mappings.
- Noul, Choice, and Score requests and the runner are code-ready. No model
  loading or GPU inference was executed by Codex.

### Noul attempt history

- The first user-executed Noul attempt on 2026-11-06 passed the clean
  dependency check and stopped during pre-load metadata validation.
- The WS1 validator incorrectly looked for `base_revision` in both files;
  official `config.json` uses `base_model_revision`, while `serving.json` uses
  `base_revision`. The validator now checks each authoritative key correctly.
- No remote code, model weights, CUDA inference, or output artifact was reached
  during this attempt.

### User-verified Noul inference

- Verified on 2026-11-06 from the user-provided native response,
  representation artifact, and execution log.
- Loaded the pinned Bosun package and Qwen base locally in BF16 on the NVIDIA
  H200, without quantization and with the PEFT adapter unmerged.
- Exposed components: official `BosunForDecision`,
  `peft.peft_model.PeftModelForCausalLM`, and
  `transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM`.
- Ordered option IDs `[false, true]` mapped to presented slots `[0, 1]` and
  decision tokens `<|decision_000|>`, `<|decision_001|>`.
- Raw logits were `[0.76171875, 2.484375]`. The official masked-softmax
  distribution was `[0.15152934193611145, 0.8484706878662109]`; its sum differs
  from one by about `3.0e-8`, within float32 conversion error.
- Native Noul exactly equals P(true), and the identity calibrated field,
  ordered option mapping, and default representation preserve the same values.
- The request used 146 input tokens and reported 1489.0 ms. This is a
  correctness check, not a performance benchmark.
- The `torch_dtype` deprecation warning did not affect loading or outputs. The
  runner now uses the equivalent `dtype` keyword for later tests.

### User-verified Choice inference

- Verified on 2026-11-06 from the locally stored native response,
  representation artifact, and execution log after the user reported success.
- Caller option order was `[billing, shipping, returns]`, while deterministic
  presentation mapped those IDs to slots `[2, 0, 1]` and decision tokens
  `[decision_002, decision_000, decision_001]`.
- The wrapper correctly restored raw logits to caller order: billing `4.9375`,
  shipping `-3.609375`, returns `-0.068359375`.
- The official distribution in caller order was billing `0.9931544065`,
  shipping `0.0001928218`, returns `0.0066527259`. Its sum differs from one by
  about `4.6e-8`, within float32 conversion error.
- Native choice and distribution argmax were both `billing`. The identity
  calibrated field, option mappings, and default representation preserved the
  same values.
- The request used 162 input tokens and reported 810.2 ms. This is a
  correctness check, not a performance benchmark.

### User-verified Score inference

- Verified on 2026-11-06 from the user-provided native response,
  representation artifact, and execution log.
- Ordered level IDs `[0, 1, 2]` mapped to presented slots `[2, 1, 0]` and were
  correctly restored to caller order.
- Caller-ordered raw logits were `[-1.359375, -0.310546875, 3.546875]`.
- The official distribution was `[0.0071949684, 0.0205366313,
  0.9722684622]`; its sum differs from one by about `6.2e-8`, within float32
  conversion error.
- The native score `1.9650735557079315` exactly equals the full-precision
  probability-weighted expected score. The identity calibrated field and
  option/slot mappings preserve the same distribution.
- The request used 154 input tokens and reported 808.8 ms. This is a
  correctness check, not a performance benchmark.

## Current gate

Kev-4B, Jeff-1.1-Qwen-4B, and Laya WS1 validation are complete. Decider is
excluded. Bosun research and isolated environment validation are complete. The
Bosun research, isolated environment, pinned downloads, local runtime, manifest
validation, and all Noul/Choice/Score smoke tests are complete. Preserve the
profiles, runtime, requests, six output artifacts, documentation, and status
record in Git before selecting the next WS1 model.
