# Bosun v3.1 1.7B WS1 profile

## Selection

WS1 selects `Hanno-Labs/bosun-v3.1-1.7b` at revision
`1d8dc82a20e4a32ed60927a47272d6efff48eed2`. The package reconstructs the
pinned `Qwen/Qwen3-1.7B` base at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, applies a rank-16 PEFT LoRA
adapter, restores the checkpoint tokenizer, and installs 256 trained decision
token rows. Up to 255 runtime candidates are supported; one slot is reserved
for null.

The source Transformers package is selected instead of GGUF because WS1 must
retain BF16-compatible trainable components and allow later LoRA work. An HTTP
server is not required.

No first-party Bosun methods paper was identified as of 2026-11-06. The
official release article and model card are the primary research sources. The
DecisionBench maintainers likewise state that its benchmark paper has not yet
been published.

## Local runtime and representations

The official local loading path is `AutoModelForCausalLM.from_pretrained()`
with `trust_remote_code=True`. The returned Bosun class preserves ordinary
causal-LM `forward()`, adds `decision_logits()` for all 256 stable slots, and
adds `predict()` for candidate-aligned typed decisions.

For each question, the runtime compiles state, instructions, type, and
candidates into the versioned prompt contract, assigns presented candidates to
stable slots, runs one forward pass, reads final-position decision-token
logits, masks unused slots, applies softmax, and maps probabilities back to
caller candidate order. Noul, Choice, and Score share this same readout.

The local adapter must preserve:

1. candidate-aligned raw logits from the valid presented slots;
2. raw masked-softmax probabilities;
3. the common-schema calibrated probability distribution;
4. option ID, decision slot, and probability mappings.

`serving.json` is authoritative. A numeric decision-token ID is only a slot;
it does not have a permanent semantic label. Representation extraction must
therefore use the per-request `presented_slot` mapping rather than assuming
token order equals option order.

## Calibration boundary

Bosun is trained against target distributions and is reported with calibration
metrics, but the release does not publish a separate post-hoc temperature or
calibration transform. Its official probability is masked softmax over valid
decision-token logits. In the universal WS1 artifact,
`calibrated_probability_distribution` is therefore an identity copy of that
official distribution, while `raw_logits` remains distinct.

This naming keeps the cross-model schema stable; it is not evidence that the
probabilities are calibrated for the eventual routing domain. Domain ECE,
NLL, Brier score, and any fitted calibration transform must be evaluated
separately before routing experiments.

## Fine-tuning

The release documents a 130,000-row, three-epoch distribution-target training
run, selected at epoch two, with rank/alpha 16/32 LoRA, batch size 32, two
gradient-accumulation steps, and seed 20260918. It publishes the adapter,
decision-token rows, loader, tokenizer, and serving contract.

No official Bosun training script or complete training corpus is public.
Consequently, the method can be reimplemented technically using the pinned
Qwen base, LoRA, trainable decision-token rows, candidate-slot randomization,
masked softmax, and a distribution loss, but the released checkpoint cannot be
reproduced exactly from public artifacts. WS1 records that limitation and does
not implement fine-tuning.

## Environment policy

The isolated `bosun` Conda environment uses Python 3.12, PyTorch CUDA 13.0,
the checkpoint-declared Transformers 4.57.0, PEFT, Safetensors, Accelerate,
and NumPy. Hugging Face Hub is pinned to 0.36.0 because Transformers 4.57.0
requires a Hub release in `[0.34.0, 1.0)`. The model and its pinned base will
both live under the project-local Hugging Face cache. Offline inference will
be enabled only after both snapshots and the remote-code files have been
validated locally.

## User-verified download

On 2026-11-06, the Bosun package resolved to
`1d8dc82a20e4a32ed60927a47272d6efff48eed2` and the Qwen base resolved to
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. All required adapter, decision
embedding, tokenizer, remote-code, serving-contract, manifest, configuration,
and base-weight files were present. The two Qwen Safetensors shards contain
3,441,185,608 and 622,329,984 bytes. The shared project Hugging Face cache was
23 GiB after download.

Snapshot directories are symlink trees into the shared Hub blob store, so a
plain `du` reports only their link metadata. This does not indicate missing
weights.

## WS1 runtime implementation

`BosunLocalRuntime` validates both snapshot revisions and every SHA-256 entry
in the checkpoint's `manifest.json` before Transformers imports the reviewed
remote model class. Loading uses `local_files_only=True`, BF16, the unmerged
PEFT adapter, and the original Qwen weights.

Inference calls the official `predict()` method. Its slot-order logits and
probabilities are mapped back through `candidate_to_slot`, while the complete
slot mapping is retained for audit. The common calibrated field is an explicit
identity copy of the official masked-softmax distribution. Noul, Choice, and
Score smoke requests are prepared but have not been executed by Codex.

## User-verified Noul result

On 2026-11-06, the pinned artifacts loaded locally in BF16 on the NVIDIA H200
with the PEFT adapter unmerged. The exposed components were the official
`BosunForDecision`, `PeftModelForCausalLM`, and `Qwen3ForCausalLM`.

The ordered IDs were `false`, `true`, mapped to decision slots zero and one.
Raw logits were `0.76171875` and `2.484375`; official masked-softmax
probabilities were `0.15152934193611145` and `0.8484706878662109`. The native
Noul value exactly matched P(true). The calibrated field intentionally equals
the official distribution because the release defines no post-hoc transform.

The request used 146 input tokens and reported 1489.0 ms. This is a correctness
test rather than a performance benchmark. Transformers emitted a deprecation
warning for the `torch_dtype` keyword; the runner now passes the equivalent
`dtype` keyword for subsequent tests. The warning did not affect this result.

## User-verified Choice result

The three caller-ordered IDs were `billing`, `shipping`, and `returns`, while
the deterministic presentation placed them in slots 2, 0, and 1. The retained
slot map correctly restored raw logits `4.9375`, `-3.609375`, and
`-0.068359375` to caller order.

The official distribution in that same order was `0.9931544065475464`,
`0.00019282178254798055`, and `0.006652725860476494`; the native selected ID
and distribution argmax were both `billing`. The request used 162 input tokens
and reported 810.2 ms. No deprecated-dtype warning appeared after the runner
update.

## User-verified Score result

The ordered level IDs `0`, `1`, and `2` were presented in slots 2, 1, and 0.
The mapping correctly restored caller-ordered logits `-1.359375`,
`-0.310546875`, and `3.546875`, and probabilities
`0.007194968406111002`, `0.0205366313457489`, and
`0.9722684621810913`.

The native score `1.9650735557079315` exactly equals the full-precision
probability-weighted expected level. The request used 154 input tokens and
reported 808.8 ms. With this result, all three Bosun primitives are validated
through the same official local decision-token readout.

## Sources

- Checkpoint: <https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b>
- Release article: <https://huggingface.co/blog/Hanno-Labs/decisionbench-bosun-v3-1>
- DecisionBench models: <https://hanno-labs.github.io/decision-bench/overview/models/>
- DecisionBench repository: <https://github.com/Hanno-Labs/decision-bench>
- Base model: <https://huggingface.co/Qwen/Qwen3-1.7B>
