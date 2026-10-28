# SuJevi

SuJevi is a local research workspace for comparing, evaluating and eventually
fine-tuning OneJev and Jev-Omni. Both models run directly in Python; no HTTP
server is required.

## Repository layout

```text
src/sujevi/             Local model runners
third_party/OneJev/     Pinned official OneJev submodule
scripts/                Environment and model setup
models/                 Downloaded weights (not tracked by Git)
examples/               Minimal local inference examples
```

## Clone and set up

```bash
git clone --recurse-submodules https://github.com/p6trick/SuJevi.git
cd SuJevi
bash scripts/setup_env.sh
conda activate sujevi
python scripts/download_models.py
```

The setup script supports both Linux x86_64 CUDA systems and NVIDIA DGX Spark.
On DGX Spark it selects a CUDA-enabled ARM64 PyTorch build instead of installing
an incompatible x86 CUDA wheel.

Download only one model when needed:

```bash
python scripts/download_models.py --onejev
python scripts/download_models.py --jev-omni
```

## Local inference

```bash
python examples/onejev_local.py
python examples/jev_omni_local.py
python examples/jev_omni_image.py --image examples/cat.jpg
```

`jev_omni_image.py` runs the same three image questions as the OneJev JSON
example and prints Jev-Omni's native prediction, confidence and probability
fields without converting them to OneJev's `noul`, `choice` or `score` schema.

Run a JSON request directly, still without a web server:

```bash
python scripts/run_onejev.py examples/request.json
```

Construct each runner once and reuse it for a complete experiment. This keeps
the model resident in unified/GPU memory between rows.

## Notes

- Model checkpoints are not committed to Git.
- Jev-Omni also needs `ffmpeg` for audio input; the Conda environment installs it.
- The unified environment pins Transformers 5.17.0, which satisfies both model releases.
- Do not load both models simultaneously unless the experiment requires it; sequential loading leaves more memory for activations and training.
