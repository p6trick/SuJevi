# Laya WS1 profile

## Selection

WS1 uses the English root checkpoint `convaiinnovations/laya` at revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`. It is a 421M-parameter,
non-autoregressive typed-decision model built from ModernBERT-large and a
two-layer decision head. The model supports `noul`, `choice`, and `score` in a
single local Python forward pass. An HTTP server is not required.

The official Python runtime is pinned to Laya `0.4.24` (release commit
`9d3fcc8924ef47b71510f77e19659f97ade0eec4`). No first-party methods paper was
identified as of 2026-11-06. The code, training implementation, checkpoint,
and model card are public under Apache-2.0.

## Local runtime and representations

The supported loading path is `laya.load()` or `laya.Agent` with a local
snapshot directory. Public `predict()` output contains the typed result and a
temperature-scaled probability distribution. It rounds probabilities to four
decimal places.

The official inference path retains the option logits before decoding:

1. `Agent._forward()` returns full-precision option logits.
2. `_option_logits()` slices each row to the number of options.
3. `_decode_answers()` divides by the configured temperature and applies
   softmax.

The WS1 local adapter will use those version-pinned official components to
preserve raw logits, raw softmax probabilities, temperature-scaled
probabilities, and the option-ID mapping before the public response is rounded.
The default model representation will be the temperature-scaled distribution.

For Noul, the semantic option order is always `false`, then `true`, and the
native `noul` value is P(true). Choice follows criteria insertion order. Score
uses zero-based level indices and returns an expected score as well as the
level distribution.

## Calibration caveats

The shipped checkpoint uses temperatures by question type and option-count
bucket, but the model card reports that it remains over-confident and
recommends refitting calibration on domain data. "Calibrated probability" in
WS1 therefore means the distribution produced by the checkpoint's official
temperature rule; it is not a claim of calibration on the eventual routing
dataset.

The published `choice:11+` temperature is `0.10058280825614929`, below the
runtime's allowed range `[0.5, 5.0]`. Laya clamps it to `0.5` and emits a
warning. Initial smoke tests use no more than ten choice options. Larger choice
sets require a separate validation record.

## Fine-tuning

Laya provides an official `laya-train` CLI and the Python API
`laya.train.finetune`. Both can train the encoder and decision head using RLCD
or soft cross-entropy, reserve a calibration split, fit temperatures, and save
a checkpoint that `laya.load()` can reopen. `--freeze-encoder` provides a
head-only alternative. WS1 records this capability but does not run training.

## Environment policy

The `laya` Conda environment is isolated from Kev and Jeff. It uses Python
3.12 and the CUDA 13.0 PyTorch wheel for the NVIDIA H200. WS1 does not install
server, ONNX, TileLang, or quantization extras. Laya 0.4.24 declares `vegaml`
as a core dependency, but the selected runtime uses the native Laya checkpoint
and does not enable `Agent(vegaml=True)`.

## Checkpoint download policy

`scripts/download_laya.py` pins the exact Hub revision and mirrors the official
Agent allow-list: `rl_agent_config.json`, `model.safetensors`, `tokenizer/*`,
and `encoder/*`. The Hub repository also contains multilingual and
typed-decisions subfolders; WS1 does not download those sibling weights. The
script validates the resolved snapshot revision and required file set but does
not load the model or access the GPU.

## User-verified WS1 result

On 2026-11-06, the pinned checkpoint loaded locally on the NVIDIA H200 through
the eager BF16 Agent path. Noul, three-option Choice, and three-level Score all
preserved raw logits, raw softmax, the official temperature-scaled softmax, and
ordered option mappings. Native four-decimal probabilities matched the stored
full-precision distributions after rounding. The Score native value also
matched the probability-weighted expected level after rounding. No validated
request was truncated.

The action head returned `act_probability=1.0` for every smoke request. It is
retained in native artifacts for provenance but excluded from the default model
representation and routing signals, consistent with the official model card's
warning that the released action head has no usable signal.

## Sources

- Code: <https://github.com/NandhaKishorM/laya>
- Release: <https://github.com/NandhaKishorM/laya/releases/tag/v0.4.24>
- Checkpoint: <https://huggingface.co/convaiinnovations/laya>
- Fine-tuning: <https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md>
