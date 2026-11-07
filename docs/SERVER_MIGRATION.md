# SuJevi server migration

Status: preparation only. No environments, caches, checkpoints or results have
been deleted. WS2 implementation is paused. The migration strategy is to clone
Git and reconstruct environments and public checkpoints on the new server.
The user has chosen to clean the old server before new-server recovery. This
accepts the risk that a public artifact or dependency may later become
unavailable. Review each target, preserve the environment exports, commit this
document and confirm the remote commit before any user-directed cleanup.

## Current inventory

The user reported local and remote `main` at
`ea93442e3bee45edb11d7cf5d60a1a87dffd26cf`, remote
`https://github.com/p6trick/SuJevi.git`. Subsequent documentation edits need
their own review and push.
Native WS1 smoke outputs are tracked. `third_party/OneJev` is a submodule at
`53df4d096db65333470c714d211930f7f6cc89dd`; clone with submodules. The LFS
listing printed no files; this does not establish Git LFS history elsewhere.

`/home/irteam/data-vol1/SuJevi` resolved to `/home1/irteam/data-vol1/SuJevi`.
Reported usage was 122G for the project, including 121G for `models/`, 89G
for its Hugging Face cache, and 941M for `.vendor/`. Figures overlap and must
not be added. Hugging Face uses shared blobs and snapshot symlinks; deleting a
snapshot folder alone does not necessarily reclaim its apparent weight size.

Important Git-ignored material:

- `models/huggingface/` contains checkpoint blobs and metadata; it is locally
  under the project, but ownership and reuse must be established before any
  cleanup. Public revision availability is not guaranteed indefinitely.
- `models/OneJev-4B/` is 9.7G and `models/Jev-Omni/` is 23G. These earlier
  models are outside the completed seven-model pool but must be retained until
  their provenance and backup needs are confirmed. See the legacy checkpoint
  section below; the old download script does not pin their revisions.
- `.vendor/kev/` contains the official source at commit
  `6b719c3c3f367295f6ef336f4f751cf5ff970abc`, plus 685M in `runs/` and 131M
  in `evals/`; preserve local artifacts until reviewed. `.vendor/jeva/` is at
  `e2104e0720100f979f2cdd456a1cbbee96da0abd`, with generated untracked
  `__pycache__` and editable-install metadata. A clean source clone alone does
  not back up ignored local artifacts.
- `runs/`, `outputs/`, `wandb/`, `.env`, and local logs or downloaded datasets
  outside the repository are not necessarily covered by Git. Review their
  actual content and any external paths before declaring the backup complete.
  Dataset inventory code and WS2 data have not been created by this project.

The seven reported environments are `/opt/conda/envs/sujevi-kev` 7.3G,
`jeff` 5.9G, `laya` 5.0G, `bosun` 5.0G, `jeva` 8.8G, `d1` 7.1G and
`openjev` 5.1G. `sujevi` also exists but its purpose and size are not yet
confirmed. Other environments may belong to other projects. `/opt/conda/pkgs`
is a shared 16G package cache; do not assume its contents can be removed.
Conda hardlinks and shared storage make summed `du` values unreliable as an
estimate of reclaimed disk space.

## Environment backup on the old server

The user must execute the commands below. Raw exports may contain absolute
paths, editable project paths, private URLs, credentials and machine-specific
prefixes. They are evidence for reconstruction, not guaranteed portable
lockfiles. Transferring them is optional under the rebuild strategy, but losing
them discards useful exact old-server provenance. If retained, use private
storage outside Git. Never publish raw exports
without inspecting and sanitizing them. The temporary project-local directory
below is ignored by Git through `models/*`; it is not an off-server backup.

```bash
cd /home/irteam/data-vol1/SuJevi
umask 077
export SUJEVI_ENV_BACKUP_DIR="$(mktemp -d "$PWD/models/migration-env-XXXXXX")"
printf 'Private environment exports: %s\n' "$SUJEVI_ENV_BACKUP_DIR"

for env_name in sujevi-kev jeff laya bosun jeva d1 openjev; do
  conda env export -n "$env_name" > "$SUJEVI_ENV_BACKUP_DIR/$env_name.full.yml"
  conda env export -n "$env_name" --from-history > "$SUJEVI_ENV_BACKUP_DIR/$env_name.history.yml"
  conda list -n "$env_name" --explicit > "$SUJEVI_ENV_BACKUP_DIR/$env_name.explicit.txt"
  env -u PYTHONPATH conda run -n "$env_name" python -m pip freeze > "$SUJEVI_ENV_BACKUP_DIR/$env_name.pip-freeze.txt"
done

ls -lh "$SUJEVI_ENV_BACKUP_DIR"
```

Check command errors and ensure every file is nonempty. Review editable source
locations and private channels/credentials locally. Do not paste secrets into
logs or Git. Conda history, explicit package lists, pip freeze and the checked-in
`configs/environments/*.yaml` serve different purposes: none alone guarantees
identical CUDA wheels, OS compatibility, dependency availability or successful
reconstruction. `sujevi` is not included above until its role is established.

User-reported export receipt on 2026-11-07: all four files for each of the
seven environments were listed with nonzero sizes and owner-only file
permissions under the Git-ignored local directory
`models/migration-env-EufCkT/`. This confirms the files exist; it does not
independently validate their content, successful reproduction, or an
off-server backup. Sensitive-information review remains pending; secure
transfer is optional. The export directory is not to be added to Git.

**Reconstruction policy:** `full.yml`, `history.yml`, `explicit.txt` and
`pip-freeze.txt` are records of the old server, not instructions to clone its
environment. Do not unconditionally run `conda env create -f` on the full
export, `conda create --file` on the explicit list, or install the full pip
freeze on the new server. First inspect the new GPU, compute capability,
driver, available VRAM, OS and architecture; choose a compatible PyTorch wheel
and its CUDA runtime, then select or rebuild GPU extension packages against
that exact PyTorch/CUDA/Python ABI. Review old build identifiers, local paths,
channels and editable installs instead of copying them blindly. Recreate and
validate each environment separately; any changed dependency or backend must
be recorded as a new environment, not described as identical to the old one.

### Recorded model-specific dependencies

These are the old server's user-verified versions, not universal requirements
for an unknown new host. All seven reports used Python 3.12.16 on Linux x86_64
and an H200. The user's private exports record additional dependencies but have
not been fully audited for portability or published.

| Model / environment | PyTorch / bundled CUDA runtime | Transformers | GPU-specific or other considerations |
| --- | --- | --- | --- |
| Kev / `sujevi-kev` | 2.8.0 / CUDA 12.8 | 5.17.0 | Pinned Kev source and `uv.lock`; PEFT adapter; original run used PyTorch reference fallback where optimized Qwen kernels were absent. |
| Jeff / `jeff` | 2.13.0+cu130 / CUDA 13.0 | 5.17.0 | `jeff==1.3.0`; a C compiler was required for Triton's CUDA helper; optional optimized Qwen kernels fell back to reference implementations. |
| Laya / `laya` | 2.13.0+cu130 / CUDA 13.0 | 5.17.0 | `laya==0.4.24`, `vegaml>=0.15.0`; validated native eager BF16 autocast, without requiring an inference-only export. |
| Bosun / `bosun` | 2.13.0+cu130 / CUDA 13.0 | 4.57.0 | PEFT 0.21.0; checkpoint-supplied remote code requires the pinned Transformers version for the validated path. |
| Jeva / `jeva` | 2.9.1+cu128 / CUDA 12.8 | 5.17.0 | PEFT 0.21.2; the old accelerated path used `causal-conv1d==1.7.0`, `fla-core`/`flash-linear-attention==0.5.2`, `tilelang==0.1.15` and NVCC 12.8.93. These compiled components need matching wheels/builds and must be checked on the new GPU; a slower documented reference path exists. |
| Liquid d1 / `d1` | 2.9.1+cu128 / CUDA 12.8 | 5.17.0 | `torchvision==0.24.1+cu128` must match the selected PyTorch build; PEFT was installed for potential future work, but d1 fine-tuning is unvalidated. |
| OpenJev / `openjev` | 2.13.0+cu130 / CUDA 13.0 | 5.17.0 | Direct unquantized BF16 loading needs sufficient available VRAM; optional optimized Qwen kernels were absent and reference implementations were used. |

The old-server driver reported CUDA 13.0 compatibility; that value is not the
CUDA runtime bundled with each PyTorch wheel. `nvcc` is a toolkit compiler,
not proof of driver or wheel compatibility. Check `torch.cuda.is_available`,
device compute capability, BF16 support and model memory on the new host after
installing the selected build. Optional extension failures or fallback paths
must be reported, and kernel doctor checks do not replace model smoke tests.

## Official code revisions

Preserve checkpoint revisions from the following table and these official code
revisions whenever they remain available and compatible with the new system:

| Model | Official source / revision |
| --- | --- |
| Kev | `https://github.com/jaredpalmer/kev.git` / `6b719c3c3f367295f6ef336f4f751cf5ff970abc` |
| Jeff | `https://github.com/j-geuter/Jeff.git` / `2c446ca3249f31c83cc4a230f3aed605858da330` (installed release `1.3.0`) |
| Laya | `https://github.com/NandhaKishorM/laya.git` / `9d3fcc8924ef47b71510f77e19659f97ade0eec4` (installed release `0.4.24`) |
| Bosun | Python model code is included in its pinned checkpoint snapshot. |
| Jeva | `https://github.com/LIQIIIII/Jeva.git` / `e2104e0720100f979f2cdd456a1cbbee96da0abd` |
| Liquid d1 | Python model code is included in its pinned checkpoint snapshot. |
| OpenJev | The official helper is included in its pinned checkpoint snapshot; SuJevi's BF16 adapter is tracked in this repository. |

Record and investigate any unavailable revision or required code change; do
not silently switch to a newer `main`. Keep the old source and local changes
without separately accepting the risk of upstream artifacts becoming unavailable.

## Seven-model checkpoint inventory

| Component | Hub repository | Pinned revision |
| --- | --- | --- |
| Kev adapter/head | `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| Kev base | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Jeff checkpoint | `jgeuter/Jeff-1.1-Qwen-4B` | `5401940b5a8c959cba82187ebfa2dce5a21abc0a` |
| Laya checkpoint | `convaiinnovations/laya` | `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Bosun adapter/code | `Hanno-Labs/bosun-v3.1-1.7b` | `1d8dc82a20e4a32ed60927a47272d6efff48eed2` |
| Bosun base | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Jeva checkpoint | `LIQIIIII/Jeva-4B` | `25ca454b3522615a243c06703ba410ecf152a128` |
| Liquid d1 checkpoint/code | `LiquidAI/d1-3B` | `051bcc464b01b9f92942b364d9586b0ef5912432` |
| OpenJev checkpoint/helper | `openjev/openjev` | `b4a89f5884f3482a5e894b2e7ff01db8ea033726` |

Jeff's checkpoint includes its base shards; OpenJev's upstream Qwen base
revision is provenance and does not need a second download for its merged
checkpoint. Each checkpoint's license and possible access restrictions must be
reviewed before downloading or moving it; Jeff and OpenJev weights are
non-commercial. Git must not contain model weights or unreviewed data.

### Pinned downloads on the new server

After installing a compatible Hugging Face CLI and reviewing access and
licenses, the user can run the following from the new checkout. These are
downloads, not commands executed during the audit:

```bash
export HF_HOME="$PWD/models/huggingface"
hf download jaredpalmer/kev-4b --revision 6cfce5c2fa4b4bd64026336ab649c5ca78857d52
hf download Qwen/Qwen3.5-4B-Base --revision 1001bb4d826a52d1f399e183466143f4da7b741b
hf download jgeuter/Jeff-1.1-Qwen-4B --revision 5401940b5a8c959cba82187ebfa2dce5a21abc0a
hf download convaiinnovations/laya --revision 7b928d828b7b0e022f929d9bd2e44165aa270148
hf download Hanno-Labs/bosun-v3.1-1.7b --revision 1d8dc82a20e4a32ed60927a47272d6efff48eed2
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
hf download LIQIIIII/Jeva-4B --revision 25ca454b3522615a243c06703ba410ecf152a128
hf download LiquidAI/d1-3B --revision 051bcc464b01b9f92942b364d9586b0ef5912432
hf download openjev/openjev --revision b4a89f5884f3482a5e894b2e7ff01db8ea033726
```

### Legacy checkpoints and Kev artifacts

The old local download metadata records `OmniJev/OneJev-4B` revision
`c5ab4dae8d9e3c12cf060db2c6487c08851fed06` and
`akhilaaa3/Jev-Omni` revision `fee89b076e2263bfa8bce8d34142caa4ca733a48`.
The public Hub resolved each exact revision during this audit. The local
OneJev `model.safetensors` SHA-256 is
`c725978346bb87ff598b04b6ffe7ea89461368e9b20a0081a3c6a85d377c8f91`;
Jev-Omni `model.safetensors` is
`d78782f3b9c3302353a7d1a1b6277fa5e05d692f4a0cb2b862797026c65eabe6`.
Both match the Hub weight ETags for their pinned revisions. Jev-Omni's local
`head.pt` SHA-256 `8c81edecd733f7db327604803c14cf9ecbee53d2af055eeb09f78b64bcbf2638`
matches the local release checksum manifest; the manifest itself has not been
independently authenticated in this audit. These checks do not establish that
all other files in either local directory match upstream, nor prove future
availability. Re-download, if needed, with:

```bash
hf download OmniJev/OneJev-4B --revision c5ab4dae8d9e3c12cf060db2c6487c08851fed06
hf download akhilaaa3/Jev-Omni --revision fee89b076e2263bfa8bce8d34142caa4ca733a48
```

In `.vendor/kev`, all 2,726 files under `runs/` and `evals/` are tracked and
the two directories had no reported modifications or untracked files at the
pinned commit. No files with common checkpoint extensions were found there.
They include experiment logs and reports; they are recoverable from that Git
commit if the upstream remains available, but an extension search is not a
proof that no weights exist under an unusual filename. Check for other local
results outside these two directories separately.

### Installation completeness audit

`configs/environments/*.yaml` records per-model direct dependency versions;
the table above records Python, PyTorch and CUDA. They are environment
profiles, not executable full installation instructions. Seven complete,
validated, hardware-independent installation command sequences are **not**
currently documented. In particular, the Kev README's `uv sync --extra serve`
creates its own uv environment by default, so it is not by itself a recipe for
recreating the old `sujevi-kev` Conda environment. Jeva's pinned README gives
`conda create -n jeva -c nvidia/label/cuda-12.8.1 python=3.12 cuda-nvcc=12.8.93`
and `python -m pip install -r requirements.txt`, but that accelerated recipe
targets CUDA 12.8 and must not be assumed suitable for an unknown GPU. Finalize
model-specific commands after hardware selection, consult each pinned source
and profile, and verify all seven independently. Exact reconstruction also
depends on source/package availability and any needed platform-specific wheels.

## New-server reconstruction checklist

The new GPU, driver, CUDA support, OS, target path and Conda availability are
not yet known. Execute only after selecting a suitable host and destination.
GPU memory and driver compatibility must be checked for each model, especially
OpenJev 27B. Host driver-reported CUDA is not the same as the PyTorch runtime.

1. Clone the final verified commit including `third_party/OneJev`:

   ```bash
   git clone --recurse-submodules https://github.com/p6trick/SuJevi.git
   cd SuJevi
   git submodule status
   nvidia-smi
   conda info
   ```

2. Optionally recover and review the private environment exports. Reconstruct the seven
   environments for the new hardware individually; consult `configs/environments/<model>.yaml`,
   available exports, source code revisions and the
   installed hardware. Recreate editable source installations at the new path.
   The actual compatible installation commands must be finalized after the
   new host and the user-provided exports are known. In particular, Kev uses
   its pinned source and `uv.lock`, Jeva uses its pinned source and matched
   compiled CUDA kernels, and Jeff requires a working C compiler for Triton.
   Do not blindly install a full export or explicit package list or apply old
   absolute prefixes or editable paths.

3. Set `HF_HOME` to the project-local `models/huggingface` unless the loader
   defaults are deliberately adapted, and download the table's exact pinned
   revisions through `hf download <repo> --revision <sha>` using a compatible
   environment. `scripts/download_laya.py` and `scripts/download_bosun.py`
   pin and validate their respective artifacts. The direct loaders expect
   Hugging Face snapshot paths under the project by default; setting `HF_HOME`
   alone does not redirect every loader. `--checkpoint` is supported by the
   seven scripts, and Bosun additionally takes `--base-checkpoint`; Kev's
   base is resolved by its official loader using its cache configuration.
   Preserve symlink targets if transferring cache snapshots; copying only the
   snapshot symlinks may produce broken files.

4. Run `scripts/check_environment.py --profile <profile>` in each environment
   (`kev` for `sujevi-kev`; otherwise use its model name). `pip check` should
   be run with `PYTHONPATH` unset. For Jeva also run the official forward
   kernel doctor. Check weights and vendor source revision before inference.

5. Run the tracked `tests/smoke_tests/*_request.json` inputs through the
   matching `scripts/run_*_native.py`, using new output paths so the checked-in
   reference artifacts are not overwritten. Kev has a combined request;
   other models have separate Noul, Choice and Score requests. See each
   runner's `--help` and `docs/models/<model>.md`. Compare option IDs and
   full-precision probability distributions against `results/native_outputs`;
   record tolerances, environment, precision, backends and differences.
   BF16, kernels, hardware and batching can cause numerical differences, so
   identical bytes and latency are not guaranteed. Also retain calibration
   semantics and do not treat absent post-hoc calibration as proven.

6. Record new-server results and any differences after reconstruction. The
   user has explicitly chosen not to wait for this step before old-server
   cleanup; preserve any private artifacts still needed before deleting them.

## Absolute paths and cleanup boundary

Model runners generally derive their root from their script and support a
`--checkpoint` argument; they are not hardcoded to `/home/irteam` in executable
paths. Existing tracked smoke JSON files record old absolute checkpoint paths
as provenance, while environment configs record old interpreter/compiler paths
as historical observations. Do not rewrite those records as new-server results.
`HF_HOME` and checkpoint overrides must be reviewed individually: several
runners default explicitly to the project cache rather than `HF_HOME`.

Deletion commands are preparation only and must not be run until the user has
reviewed the targets and other users' dependencies.
Do not clean shared Conda or Hugging Face caches wholesale. Decide whether any
non-reproducible local artifacts require a private backup before cleanup.
