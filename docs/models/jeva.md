# Jeva-4B WS1 profile

The selected checkpoint is `LIQIIIII/Jeva-4B` at revision
`25ca454b3522615a243c06703ba410ecf152a128`. The official code is
`LIQIIIII/Jeva` at revision `e2104e0720100f979f2cdd456a1cbbee96da0abd`.
The model accompanies *Jeva: System One Models from First Principles*
(arXiv:2610.18773). Code and checkpoint are Apache-2.0.

The user verified local Noul, Choice and Score inference using the official
Python runtime on an NVIDIA H200 with BF16 weights and an unmerged LoRA
adapter. No HTTP server or quantization is used. The official runtime freezes
parameters for inference; later training should use its separate fine-tuning
path and clear inference caches after weight changes.

The decision head computes option scores, divides by its learned `tau_dec`,
then applies softmax. The representation artifact saves both scores and
their softmax, temperature-scaled decision logits, native probabilities,
temperature, and ordered option IDs. It does not claim the native distribution
is post-hoc calibrated: that field is null until a calibration method is
established and assessed. Native probabilities are the current default.

These smoke tests use closed-set choices. The separately saved head null
probability is not part of those distributions; it must not be mixed into
their normalization without explicitly changing the task semantics.

Official `jeva-finetune` supports head-only and LoRA modes. No fine-tuning
has been performed. The runtime and output files are detailed in
`results/native_outputs/README.md`.
