# New-server rebuild and validation

This is a preparation guide, not a claim that the unknown new host is
compatible. Keep the old server's Conda environments, checkpoints, caches,
exports, and results intact. Execute these commands on the **new** server only.
No installation, download, inference, or testing was performed while writing
this guide. Old-server reference versions and checkpoint revisions are in
[SERVER_MIGRATION.md](SERVER_MIGRATION.md).

## 1. Clone, hardware and paths

Choose the parent directory on the new server. Do not clone over an existing
checkout. The submodule is needed for the project's OneJev code, though it is
not one of these seven smoke tests.

```bash
git clone --recurse-submodules https://github.com/p6trick/SuJevi.git
cd SuJevi
export SUJEVI_ROOT="$PWD"
git rev-parse HEAD
git submodule status
conda info
nvidia-smi
```

Record Linux distribution/architecture, GPU model, free VRAM, compute
capability, NVIDIA driver, and BF16 support. `nvidia-smi`'s CUDA version is the
driver's supported level, not the installed PyTorch runtime. Consult the
PyTorch and GPU extension vendors for the actual driver, GPU architecture,
Python ABI and wheel compatibility. OpenJev's 27B BF16 checkpoint needs
substantial free VRAM beyond its approximately 54 GB weights.

Create a fresh output directory so reference files are not overwritten:

```bash
export HF_HOME="$SUJEVI_ROOT/models/huggingface"
export PYTHONPATH="$SUJEVI_ROOT/src"
export SUJEVI_RUN_DIR="$(mktemp -d "$SUJEVI_ROOT/results/new-server-XXXXXX")"
printf 'Results: %s\n' "$SUJEVI_RUN_DIR"
```

All seven runners derive their default checkpoint paths from the checkout,
not the former `/home/irteam` path. They do **not** all honor `HF_HOME` for
their checkpoint defaults. This guide deliberately sets `HF_HOME` to their
default project-local cache. For a different cache pass `--checkpoint` with
the actual pinned snapshot path to every runner; for Bosun also pass
`--base-checkpoint`. Kev resolves its base through the HF cache, so set
`HF_HOME` to the cache holding the pinned base as well. Some runners require
the snapshot directory basename to be the revision SHA. Historical absolute
paths in checked-in JSON are provenance, not executable defaults.

## 2. Clone pinned official Python sources

These destinations must not already exist; do not replace local work. Git
checkout at a commit is needed only in these newly created official clones.
Bosun and d1 supply model code in their pinned HF snapshots; OpenJev supplies
its helper there and SuJevi's direct BF16 runner is in this checkout.

```bash
mkdir -p "$SUJEVI_ROOT/.vendor"
git clone https://github.com/jaredpalmer/kev.git "$SUJEVI_ROOT/.vendor/kev"
git -C "$SUJEVI_ROOT/.vendor/kev" checkout --detach 6b719c3c3f367295f6ef336f4f751cf5ff970abc
git clone https://github.com/j-geuter/Jeff.git "$SUJEVI_ROOT/.vendor/jeff"
git -C "$SUJEVI_ROOT/.vendor/jeff" checkout --detach 2c446ca3249f31c83cc4a230f3aed605858da330
git clone https://github.com/NandhaKishorM/laya.git "$SUJEVI_ROOT/.vendor/laya"
git -C "$SUJEVI_ROOT/.vendor/laya" checkout --detach 9d3fcc8924ef47b71510f77e19659f97ade0eec4
git clone https://github.com/LIQIIIII/Jeva.git "$SUJEVI_ROOT/.vendor/jeva"
git -C "$SUJEVI_ROOT/.vendor/jeva" checkout --detach e2104e0720100f979f2cdd456a1cbbee96da0abd
```

Verify each with `git -C "$SUJEVI_ROOT/.vendor/<name>" rev-parse HEAD` and
inspect upstream installation notes. The official revisions are fixed; do not
silently substitute a different version if dependencies are incompatible.

## 3. Create isolated environments and select GPU builds

Run on the new host after checking that these environment names do not already
belong to another project. Python 3.12.16 is the old-server reference; check
availability for the new OS and architecture. No Conda export is a portable
lockfile.

```bash
conda env list
conda create -n sujevi-kev python=3.12.16 pip
conda create -n jeff python=3.12.16 pip
conda create -n laya python=3.12.16 pip
conda create -n bosun python=3.12.16 pip
conda create -n jeva python=3.12.16 pip
conda create -n d1 python=3.12.16 pip
conda create -n openjev python=3.12.16 pip
```

**Hardware gate:** In each activated environment, first install a PyTorch
build selected for the new GPU/driver and matching that model's version in
`configs/environments/<profile>.yaml` where compatible. For d1 also select a
matching torchvision build. The old CUDA 12.8/13.0 wheels are observations,
not choices for this host. If no compatible build satisfies a pinned source or
profile, stop and record the incompatibility; changing versions is a new,
unvalidated configuration. Jeva CUDA extensions and compiler/toolkit are also
hardware- and PyTorch-ABI-dependent: choose compatible builds after this gate,
or use the official reference path and report the backend. Jeff's first
request may need a working C compiler for Triton.

For each environment after selecting PyTorch, create a pip constraint from the
**installed** torch (and, for d1, torchvision) so subsequent dependency
resolution does not silently replace it. Run this block again immediately after
each `conda activate` below; do not reuse another environment's constraint.

```bash
export SUJEVI_TORCH_CONSTRAINT="$(mktemp)"
env -u PYTHONPATH python -m pip freeze | grep -E '^(torch|torchvision)==' > "$SUJEVI_TORCH_CONSTRAINT"
test -s "$SUJEVI_TORCH_CONSTRAINT" && cat "$SUJEVI_TORCH_CONSTRAINT"
```

Stop if the file does not contain the expected selected `torch` version, or
for d1 also the expected `torchvision`. Package resolution may fail if the
selected wheel and the source's requirements conflict; do not bypass the
constraint to hide the conflict. The following are preparation recipes based
on old profiles, **not yet validated on the new server**. Review resolver
output and source requirements; package availability can change.

### Kev — `sujevi-kev`

Activate `sujevi-kev`, install its GPU-compatible PyTorch, generate the
constraint block, then run:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" -e "$SUJEVI_ROOT/.vendor/kev[serve]"
```

This installs from the pinned code, not necessarily the exact transitive
versions in `uv.lock`. The upstream `uv sync --extra serve` typically manages
its own environment and is not automatically equivalent to this Conda
installation; retain the lock for comparison. Original profile: torch 2.8.0,
Transformers 5.17.0; check resolver results against `kev.yaml`.

### Jeff — `jeff`

Activate `jeff`, select PyTorch, generate the constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" -e "$SUJEVI_ROOT/.vendor/jeff" \
  'transformers==5.17.0' 'safetensors==0.8.0' 'huggingface_hub==1.32.0' 'accelerate==1.15.0'
```

Check `jeff` reports version 1.3.0 and that a compatible C compiler is
available. Compiler/toolkit installation is host-dependent.

### Laya — `laya`

Activate `laya`, select PyTorch, generate the constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" -e "$SUJEVI_ROOT/.vendor/laya" \
  'transformers==5.17.0' 'safetensors==0.8.0' 'huggingface_hub==1.32.0' 'numpy==2.5.3' 'vegaml>=0.15.0'
```

Check `laya` reports version 0.4.24. Its released `choice:11+` temperature
warning must remain visible and is not evidence of calibration.

### Bosun — `bosun`

Activate `bosun`, select PyTorch, generate the constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" 'transformers==4.57.0' \
  'peft==0.21.0' 'safetensors==0.8.0' 'huggingface_hub==0.36.0' \
  'accelerate==1.15.0' 'numpy==2.5.3'
```

Its model uses checkpoint-supplied remote code. Review that code before use.

### Jeva — `jeva`

Activate `jeva`, select PyTorch, generate the constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" -e "$SUJEVI_ROOT/.vendor/jeva"
```

The source requires torch 2.9.1; its optional old CUDA 12.8 accelerated
requirements must not be installed blindly. Choose and document a compatible
kernel path after checking the host. For accelerated builds, inspect pinned
`requirements.txt`, `requirements-core.txt`, and the official doctor; the
old H200 kernel versions are in `configs/environments/jeva.yaml`.

### Liquid d1 — `d1`

Activate `d1`, select matching PyTorch and torchvision, generate the
constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" 'transformers==5.17.0' \
  'peft==0.21.2' 'accelerate==1.12.0' 'huggingface_hub>=1.5,<2' \
  'safetensors>=0.8,<1' 'pillow>=10,<13'
```

The checkpoint executes its supplied remote code; review it before use.

### OpenJev — `openjev`

Activate `openjev`, select PyTorch, generate the constraint, then:

```bash
python -m pip install -c "$SUJEVI_TORCH_CONSTRAINT" 'transformers==5.17.0' \
  'peft==0.21.0' 'accelerate==1.15.0' 'huggingface_hub==1.32.0' \
  'safetensors==0.8.0' 'pillow>=10,<13'
```

This is SuJevi's direct BF16 runtime, not a verified reproduction of the
official FP8 server. Helper prompt/server parity remains unverified.

### Verify each environment

Activate each environment in turn and use its profile (`kev` for
`sujevi-kev`; the other profiles have the same name as the environment):

```bash
env -u PYTHONPATH python -m pip check
env -u PYTHONPATH python "$SUJEVI_ROOT/scripts/check_environment.py" --profile <PROFILE>
```

Replace `<PROFILE>` explicitly; it is not shell syntax. The old-server profile
checker can report a mismatch for a deliberate new CUDA/runtime selection;
inspect and record each failure rather than asserting the new host is invalid
solely for not matching the old runtime. For Jeva, run
`jeva-doctor --check-kernels --strict` only when testing the accelerated
kernel path; it executes GPU kernels and is not needed for the reference path.

## 4. Download pinned model snapshots

Use an environment with a compatible `hf` CLI and sufficient disk space.
Follow the exact nine `hf download` commands in the **Pinned downloads on the
new server** section of [SERVER_MIGRATION.md](SERVER_MIGRATION.md), with
`HF_HOME` set above. These cover the seven checkpoints plus Kev and Bosun base
checkpoints. Check access, licenses, snapshot directory names and file
integrity; existing downloads do not prove full integrity. Then:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Offline mode prevents an inference run from quietly retrieving another
revision; verify the pinned files and pinned base are complete first. HF
snapshots may be symlinks to blobs: do not move snapshot directories alone.

## 5. Smoke tests

Each command below loads a model and uses GPU memory. Run only after checking
free VRAM and other users' processes; do not run them concurrently. The shell
must have `SUJEVI_ROOT`, `SUJEVI_RUN_DIR`, `HF_HOME` and `PYTHONPATH` set as
above. Use a fresh `SUJEVI_RUN_DIR` for every run. Do not use `--overwrite`.

Kev's one request contains Noul, Choice and Score:

```bash
conda activate sujevi-kev
python "$SUJEVI_ROOT/scripts/run_kev_native.py" \
  --request "$SUJEVI_ROOT/tests/smoke_tests/kev_native_request.json" \
  --native-output "$SUJEVI_RUN_DIR/kev.json" \
  --probabilities-output "$SUJEVI_RUN_DIR/kev.probabilities.json"
```

For each of the other six models, activate the named environment, then run
the following block with `MODEL` set to `jeff`, `laya`, `bosun`, `jeva`,
`d1`, and `openjev` respectively. For example, use `conda activate jeff`
and `MODEL=jeff` first. Run the block separately for each model:

```bash
MODEL=jeff
for kind in noul choice score; do
  python "$SUJEVI_ROOT/scripts/run_${MODEL}_native.py" \
    --request "$SUJEVI_ROOT/tests/smoke_tests/${MODEL}_${kind}_request.json" \
    --native-output "$SUJEVI_RUN_DIR/${MODEL}.${kind}.json" \
    --representations-output "$SUJEVI_RUN_DIR/${MODEL}.${kind}.representations.json" || break
done
```

Explicit activation order: `conda activate jeff`, `conda activate laya`,
`conda activate bosun`, `conda activate jeva`, `conda activate d1`, and
`conda activate openjev`; set `MODEL` to the matching name each time. These
commands use the pinned default cache path. Bosun needs its separate pinned
base snapshot too. Inference success alone does not establish statistical
calibration, and BF16/kernel differences can change numeric results. Preserve
native versus calibrated semantics, option ordering and truncation warnings.

## 6. New-server verification record

Copy this checklist to a private experiment record, fill in actual versions,
paths, timestamps, commands, warnings, failures and result-file locations.
Do not edit the historical checked-in outputs to represent new results.

- [ ] Git and submodule commits recorded; official source revisions verified.
- [ ] GPU/driver/compute capability, free VRAM, Python, PyTorch/CUDA runtime,
      Transformers, precision and GPU extensions/backends recorded per model.
- [ ] All nine snapshot revisions and file integrity checked; licenses reviewed.
- [ ] `pip check` and environment profile checked; explain all differences.
- [ ] Kev: Noul / Choice / Score; native output and probability artifact.
- [ ] Jeff: Noul / Choice / Score; raw and released distributions.
- [ ] Laya: Noul / Choice / Score; temperature bucket and warnings.
- [ ] Bosun: Noul / Choice / Score; base revision, slot mapping, identity calibration.
- [ ] Jeva: Noul / Choice / Score; native/null semantics and kernel backend.
- [ ] d1: Noul / Choice / Score; token-group readout and native probabilities.
- [ ] OpenJev: Noul / Choice / Score; BF16, helper/server parity limitation.
- [ ] Option IDs/order and full-precision probabilities reviewed against the
      tracked references; comparison tolerances and deviations recorded.
- [ ] Any failure, unsupported operation, fallback or hardware limitation is
      recorded rather than marked as a pass.
