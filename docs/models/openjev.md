# OpenJev 27B WS1 profile

The selected BF16 checkpoint is `openjev/openjev` at revision
`b4a89f5884f3482a5e894b2e7ff01db8ea033726`. Its `NOTICE` identifies the base
as `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The weights are CC BY-NC 4.0;
helper and serving code are Apache-2.0. Commercial use of the weights requires
a separate license.

The user verified local text Noul, Choice and Score smoke inference on an
NVIDIA H200 with unquantized BF16 weights and direct Python loading. The
SuJevi adapter implements the one-question text prompt and targeted letter
token readout of the supplied helper. It does not invoke the official helper
server. Numerical and prompt parity against the official vLLM/FP8 server
have not been verified. Image input, multi-question requests, more than 52
options and fine-tuning are not validated by this adapter.

Each representation preserves option IDs, option letters and token IDs, raw
logits, raw softmax and the helper's temperature-scaled distribution.
Choice and Score apply `T=0.85`. Noul first applies that same temperature,
then applies the helper's `T=1.829074`, bias 0 and probability clamp. The
calibrated field means the official helper transform was applied, not that
calibration has been demonstrated on the research routing dataset. Native-
shaped outputs are rounded; representations preserve full precision.

The official measured recipe uses vLLM with online FP8 quantization, so its
performance figures are not applicable to this BF16 direct runtime. In the
user's smoke runs the optional causal-convolution and flash-linear-attention
kernels were absent and Transformers used its slower reference path.

Checkpoint checksum verification has not been reported. Official training
code and LoRA compatibility remain unverified; WS1 has not implemented or
performed fine-tuning.

Source: <https://huggingface.co/openjev/openjev>.
