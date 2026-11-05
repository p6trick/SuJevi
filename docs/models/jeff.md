# Jeff-1.1-Qwen-4B local runtime

## Selected release

- Checkpoint: `jgeuter/Jeff-1.1-Qwen-4B`
- Checkpoint revision: `5401940b5a8c959cba82187ebfa2dce5a21abc0a`
- Official code: `j-geuter/Jeff`, release `v1.3.0`
- Code commit: `2c446ca3249f31c83cc4a230f3aed605858da330`
- Base model: `Qwen/Qwen3.5-4B` at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Checkpoint license: CC BY-NC 4.0
- Code and base-model license: Apache-2.0

Jeff 1.1 is the official successor to the models reported in *Releasing the
Inner Jev: A Simple Recipe for Open Decision Models* (arXiv:2610.17205v1). It
uses the paper's LoRA-plus-residual-decision-head method, with additional data,
800 training steps, a three-run model soup, and bidirectional attention. The
current paper version reports Jeff 1.0; the model card is the primary record of
the 1.1 additions.

## Local loading

SuJevi uses the official `jeff.JeffModel` class directly. It does not start the
official HTTP server. The checkpoint is loaded locally in BF16 without
quantization, with reasoning disabled and `commit=False`. Disabling commitment
keeps the calibrated Noul probability rather than moving it to a JevBench
abstention-band edge.

The official PyTorch 2.13 runtime dispatches some Qwen operations through
Triton. Triton compiles a small CUDA helper on its first request, so the Conda
environment must expose a C compiler through `CC` or a standard `cc`/`gcc`
executable. SuJevi's environment checker treats this as a required runtime
prerequisite.

The wrapper exposes:

- `runtime.backbone`: released Qwen causal language model;
- `runtime.base_model`: Qwen text backbone used by the decision forward pass;
- `runtime.decision_head`: fp32 residual decision head;
- `runtime.tokenizer`; and
- `runtime.serving_rule`: per-question-type calibration temperatures.

The generic Hugging Face `AutoModelForCausalLM` example is insufficient for
decision inference because it does not load `head.safetensors`, apply Jeff's
bidirectional attention configuration, or apply `serving_rule.json`.

## Representation contract

Each question is evaluated once. From that same forward pass SuJevi retains:

1. `raw_logits`: decision-head logits before temperature scaling;
2. `raw_probability_distribution`: softmax of the raw logits;
3. `calibrated_probability_distribution`: the official per-type temperature
   applied with Noul commitment disabled; and
4. `option_id_probability_mapping`: the calibrated probabilities paired with
   option IDs in prompt order.

`calibrated_probability_distribution` is the default model representation.
For Noul, both `true` and `false` are stored even though the native Jeff answer
contains only `P(true)`. Choice keeps request criterion order. Score keeps the
ordered level IDs `0` through `K-1`.

The raw-logit representation contract currently covers the decision head's
native capacity of at most 16 options. Jeff can serve up to 256 options through
hierarchical rounds, but those rounds do not produce one directly comparable
raw-logit vector and are outside the WS1 smoke gate.

## Fine-tuning status

The official repository provides serving and component-loading code but no
training entrypoint. The paper describes the LoRA and decision-head recipe,
but WS1 does not implement it. Exact Jeff 1.1 reproduction is also not possible
from the currently public artifacts because its full data construction and
training pipeline are not released.

## Execution gates

The user executes all state-changing or GPU commands. Gates are performed in
this order:

1. create and validate the isolated `jeff` Conda environment;
2. download the pinned checkpoint revision;
3. run the Noul smoke request;
4. run the Choice smoke request; and
5. run the Score smoke request.

No later gate is marked successful before its user-provided output is reviewed.

## Validated runtime

User-executed H200 smoke tests successfully loaded the pinned BF16 checkpoint
through the official `JeffModel` and verified Noul, Choice, and Score. Every
artifact contains raw logits, raw softmax probabilities, calibrated
probabilities, and ordered option-ID mappings. The native answer agrees with
the calibrated representation for every type.

Transformers used its correct reference PyTorch paths because the optional
`causal_conv1d` and `flash-linear-attention` packages were not installed. This
is acceptable for the WS1 correctness gate. Kernel optimization should remain
a separate measured change before throughput experiments.
