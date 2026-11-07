# WS2 text dataset investigation

Status: research only, 2026-11-07. No datasets downloaded or split; no dataset
code executed. The intended model pool contains Kev, Jeff, Laya, Bosun, Jeva,
Liquid d1 and OpenJev. The common model output interface is not yet implemented.

## S1MB English v1

- Official dataset: https://huggingface.co/datasets/hotchpotch/S1MB-english-v1
- Suggested snapshot to pin after confirmation:
  `10bcb8c6916b98b966cc1f03aab1781a7c354b4c`.
- Published evaluation-only `test` split, 106 active subset configurations,
  14,009 cases and 26,269 individual judgments. One case may contain multiple
  decisions. The 137 subset-by-type benchmarks comprise 59 Noul, 57 Choice and
  21 Score benchmarks; these are not judgment counts.
- A published complete S1MB result reports 12,357 Noul, 8,764 Choice and
  5,148 Score successful judgments, totaling 26,269. Those are evaluation-run
  counts; verify against the exact pinned dataset revision before treating
  them as that snapshot's authoritative counts:
  https://huggingface.co/datasets/hotchpotch/s1mb-result/discussions/40
- Input fields include `input.state_json`, `input.decisions`, `case_id`,
  `group_id`, `input_hash`, `language`, and `targets`. Targets align to
  `decision_id` and may include distributions. Parse JSON-valued text rather
  than passing its quoted representation blindly; keep targets and
  provenance out of model inputs. Preserve authored Noul false/true meanings,
  choice IDs/order/descriptions, and Score values.
- Select an explicit `load_dataset` config and the `test` split. The official
  card example uses `laya__typed_decisions`; it is one config, not all 106.
  `training-manifest.json` identifies active test membership despite its
  name. Do not include quarantined sources from other manifests.
- The dataset is a collection of upstream sources with mixed licenses, some
  unknown or not separately declared. Review source-specific reuse and
  redistribution terms; this release does not relicense them.
- The six synthetic generalization benchmarks have 100 cases each and
  author/self-reviewed labels, not independently human-validated labels.
  S1MB does not establish unseen-task or model-training non-overlap.

## DecisionBench

- Official dataset: https://huggingface.co/datasets/Hanno-Labs/decision-bench
- Suggested snapshot to pin after confirmation:
  `071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb`.
- Canonical `eval` split: 23,900 rows / distinct row IDs; 43 task IDs, 27
  families, 28 domains and 2–255 candidates. Three primitives map to Noul
  (`binary_classification`), Choice (`candidate_selection`), and Score
  (`ordinal_scoring`). Exact counts across the full set by primitive are not
  stated in the dataset card and need to be counted on the pinned snapshot.
  Its 1,200 reasoning rows include 400 of each primitive; these are not the
  counts for the full dataset.
- Row fields include `row_id`, `task_id`, `primitive`, `family`, `domain`,
  `instruction`, `state_json`, `candidates_json`, `gold_candidate_id`,
  `gold_probabilities` and `source_json`. Candidate IDs/order, labels,
  descriptions and ordinal values must be aligned. Compact fields are
  alternative input renderings; use the standard fields unless a separate
  explicitly labeled protocol is chosen.
- Dataset card declares license `other`; check source-level terms in the
  manifests and provenance. Bosun is developed by DecisionBench's publisher;
  a separate dataset does not prove unseen inputs or unseen model training.

## Proposed three-way S1MB partition

The requested Clustering, Model Representation and In-Dataset Test sets are
derived research partitions of the released test-only collection, not
official S1MB train/validation/test splits and not directly comparable to the
full official benchmark score. Do not use S1MB for model fine-tuning without
separately checking the release policy and source terms. Use active subset
membership and review licenses before committing to reuse.

Split by connected groups, never by individual judgments: all decisions in a
case and all cases in the same `group_id` must stay together; additionally join
duplicate or near-duplicate inputs and linked upstream source families when
detected. Preserve contextual families, query/document groups and source
relations. Inspect group-ID scope and missing IDs instead of assuming global
uniqueness. Consider a strict task/source-held-out variant in addition to the
requested within-dataset split; group splitting alone does not ensure task or
domain novelty.

Fix the group-aware assignment, seed and ratios before looking at test labels
or test performance. Stratify approximately by subset and decision type while
retaining enough groups and judgments per partition; record achieved counts,
shortfalls, cases, judgments, labels, option counts and source distributions.
Keep a separately held-out calibration subset if fitting calibration; never fit
temperatures on the in-dataset or DecisionBench test. Do not require a fixed
ratio before the group audit establishes feasible sizes.

Treat DecisionBench as a separate cross-dataset test, not automatically as
unseen to the pretrained models. Audit S1MB/DecisionBench exact and near
overlap and available model-training provenance, report unknowns, and freeze
test access before choosing methods. Dataset name differences alone do not
establish independence. Reconcile input lengths and candidate limits across
all models, track exclusions and failure coverage, and preserve a common
eligible subset for direct comparisons.

## Reproducible acquisition

After source terms and snapshot choices are confirmed, a user can download
the pinned repositories with `hf download --repo-type dataset --revision` to
the project-local Hugging Face cache. The S1MB Hub repository is approximately
115 MB and DecisionBench approximately 976 MB as reported by their Hub file
views; actual disk/cache requirements may differ. Record snapshot SHA and
checksums. Dataset installation, download, counting and splitting must be run
by the user, not Codex.

Sources: https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark ;
https://github.com/hotchpotch/S1MB/blob/main/docs/benchmark_scope.md ;
https://huggingface.co/datasets/hotchpotch/S1MB-english-v1 ;
https://huggingface.co/datasets/Hanno-Labs/decision-bench
