# Native model outputs

This directory stores responses exactly in each model's native schema. Do not
replace native probability fields with the future universal routing interface.

`kev-4b.json` is created by `scripts/run_kev_native.py` from the official Kev
Python loader, without an HTTP server. `kev-4b.probabilities.json` keeps the
complete distribution for every question, including both `false` and `true`
for Noul. Neither is considered successful until the user-provided execution
log is reviewed.

Jeff smoke tests create one native response and one representation artifact per
question type. A Jeff representation artifact retains the raw decision-head
logits, raw softmax, calibrated distribution, and ordered option-ID mapping.
The calibrated distribution is the default representation. No Jeff output is
considered successful until the corresponding user execution is reviewed.

The user-verified Jeff artifacts are:

- `jeff-1.1-qwen-4b.noul.json` and
  `jeff-1.1-qwen-4b.noul.representations.json`;
- `jeff-1.1-qwen-4b.choice.json` and
  `jeff-1.1-qwen-4b.choice.representations.json`; and
- `jeff-1.1-qwen-4b.score.json` and
  `jeff-1.1-qwen-4b.score.representations.json`.

Laya smoke tests use the official eager `laya.Agent` forward and decoder. Each
representation artifact retains full-precision raw logits, raw softmax,
temperature-scaled softmax, the ordered option-ID mapping, and the applied
temperature bucket. Native responses keep the official four-decimal answer
format and action field. The action probability is preserved for provenance
but is not a routing representation.

The user-verified Laya artifacts currently are:

- `laya.noul.json` and `laya.noul.representations.json`; and
- `laya.choice.json` and `laya.choice.representations.json`; and
- `laya.score.json` and `laya.score.representations.json`.

Bosun smoke tests call the checkpoint-provided `BosunForDecision.predict`
method directly. Each representation artifact retains caller-aligned raw
logits, official masked-softmax probabilities, an identity calibrated field,
the candidate-to-slot map, slot-order logits/probabilities, decision-token
IDs, and option-ID mappings. The checkpoint manifest is verified before the
reviewed remote code is imported. No Bosun output is considered successful
until the corresponding user execution is reviewed.

The user-verified Bosun artifacts currently are:

- `bosun-v3.1-1.7b.noul.json` and
  `bosun-v3.1-1.7b.noul.representations.json`; and
- `bosun-v3.1-1.7b.choice.json` and
  `bosun-v3.1-1.7b.choice.representations.json`; and
- `bosun-v3.1-1.7b.score.json` and
  `bosun-v3.1-1.7b.score.representations.json`.

Jeva's user-verified Noul, Choice, and Score artifacts use the names
`jeva-4b.<type>.json` and `jeva-4b.<type>.representations.json`. Each
representation preserves pre-temperature scores and softmax, learned
temperature, decision logits, native probabilities, and option mappings.
No post-hoc calibration transform has been established: the calibrated
field is null and the native distribution is the current default. The
separately stored head null probability is not included in the closed-set
Noul, Choice, or Score distributions used in these tests.

Liquid d1's user-verified text Noul, Choice and Score artifacts are named
`d1-3b.<type>.json` and `d1-3b.<type>.representations.json`. Representations
preserve the official option-token groups and logits, max-pooled option
logits, raw softmax, native probabilities and option-ID mapping. The default
official runtime uses no post-hoc calibration object, so the calibrated field
is null; native probabilities are the current default, without asserting
calibration on the routing dataset.

OpenJev's user-verified text Noul, Choice and Score artifacts are named
`openjev-27b.<type>.json` and `openjev-27b.<type>.representations.json`.
These responses come from a SuJevi direct-BF16 Python adaptation of the
official helper, not from the official vLLM server. Representation artifacts
retain raw logits, raw probabilities, helper-temperature probabilities,
calibrated probabilities and option/token mappings. For Noul, the official
yes/no slope is applied after the shared readout temperature. Parity with the
official vLLM/FP8 server and calibration quality on routing data remain
unverified. The helper's public-shaped responses round to four decimals.
