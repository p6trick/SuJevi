# Kev-4B local runtime and fine-tuning compatibility

## Loading strategy

SuJevi loads the pinned local Kev-4B snapshot with the official
`kev.checkpoint.Checkpoint` loader. The loader reconstructs all released model
components:

- `Qwen/Qwen3.5-4B-Base` at the revision recorded in `head.pt`;
- the rank-16 PEFT LoRA adapter from `adapter_model.safetensors`;
- Kev's pointer decision head and fitted calibration temperature from
  `head.pt`; and
- the released tokenizer and special tokens.

`src/sujevi/kev_local.py` sets `merge=False`, `backend="torch"`, and uses BF16
by default. It performs no quantization and does not start an HTTP server. The
runtime exposes:

- `runtime.model`: official `DecisionModel`;
- `runtime.adapter_model`: unmerged PEFT model;
- `runtime.backbone`: underlying Qwen backbone;
- `runtime.decision_head`: pointer head;
- `runtime.tokenizer`; and
- `runtime.metadata`: checkpoint base revision, LoRA rank, temperature, and
  other official metadata.

The native response retains Kev's Noul, Choice, and Score schema. A second
artifact records the complete probability vector for every question. This is
important because the native Noul response reports only `p(true)`, while the
direct model output contains `[p(false), p(true)]`.

## Fine-tuning support

Fine-tuning is officially supported. `python -m kev.train` trains the LoRA
adapter and pointer head while leaving the Qwen backbone frozen. For an
existing release, the intended continuation path is `--init_from` pointing to
the Kev checkpoint; this warm-starts both adapter and head. Kev also supports
full-backbone training with `--full_ft 1`, BF16 weights, fp32 optimizer masters,
and multi-GPU FSDP2, but that is materially more expensive.

For the planned surrogate adaptation, LoRA continuation from the pinned local
Kev-4B checkpoint is the preferred first experiment. The direct inference
runtime is intentionally component-accessible, but training should be launched
through the official trainer so its data conversion, losses, calibration,
checkpoint metadata, and saving behavior remain intact.

No fine-tuning command has been executed, and no fine-tuning implementation is
added at this stage.

## Validated runtime

The user-verified H200 run loaded the adapter unmerged in BF16 and exposed the
expected PEFT model, Qwen3.5 text backbone, and Kev pointer head. Noul, Choice,
and Score all returned normalized full distributions with the released
calibration temperature.

Transformers reported that `causal_conv1d` and `flash-linear-attention` were
not installed and selected its reference PyTorch implementations. Those paths
are correct but slower. They are acceptable for the WS1 functional gate; any
optimized-kernel installation should be treated as a separate, measured
environment change before performance or fine-tuning experiments.
