# WS1 model status

Last updated: 2026-11-05 (Asia/Seoul)

Status values are `Pending`, `Code Ready`, `Awaiting User Execution`,
`Success`, `Failed`, and `Blocked`. A step is marked `Success` only from a
user-provided execution result.

| Order | Model | Research | Environment | Download | Native inference | Noul/Choice/Score |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kev-4B | Code Ready | Awaiting User Execution | Pending | Pending | Pending |
| 2 | Jeff | Pending | Pending | Pending | Pending | Pending |
| 3 | Decider | Pending | Pending | Pending | Pending | Pending |
| 4 | Laya | Pending | Pending | Pending | Pending | Pending |
| 5 | Bosun | Pending | Pending | Pending | Pending | Pending |

## Kev-4B research record

- Official code: `jaredpalmer/kev`, release `kev-1.0`, pinned to
  `6b719c3c3f367295f6ef336f4f751cf5ff970abc`.
- Checkpoint: `jaredpalmer/kev-4b@v1.0`, currently resolving to
  `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`. The model card identifies
  `139fdd94` as the released weights revision.
- Base: `Qwen/Qwen3.5-4B-Base`, pinned to
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Runtime: official Kev source with its committed `uv.lock` and `serve` extra.
- Native API: TypeSafe-compatible `POST /v1/systemone`.
- Native question types: `noul`, `choice`, and `score`.
- Calibration: checkpoint temperature is applied by default. Native option
  distributions must be retained without conversion during WS1 validation.
- License: Apache-2.0 for Kev and the Qwen base.
- Hardware target: NVIDIA H200. No environment, CUDA, weight, or inference
  command has yet been executed by Codex or confirmed by the user.

## Current gate

Create the isolated Kev environment and run `scripts/check_environment.py`.
Do not download model weights until the environment result is reviewed.
