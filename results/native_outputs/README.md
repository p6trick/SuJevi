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
