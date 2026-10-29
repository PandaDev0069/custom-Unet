# Executed validation

## PQ validation performance update (2026-10-29)

- CUDA suite with native probing: **40 passed, 1 two-GPU-only test skipped**
  ([XML](../validation_runs/pq_speed_gpu_tests.xml)). CPU suite: **37 passed,
  4 CUDA-only skipped** ([XML](../validation_runs/pq_speed_cpu_tests.xml)).
- Exact label-count/RLE parity covers overlapping/disconnected GT, strict 0.5
  threshold, empty cases and randomized fragmented predictions. A guard verifies
  the fast path does not materialize one dense mask per predicted instance.
- First/every-second/final PQ cadence and unchanged every-epoch validation loss
  pass regression checks. After including distributed aggregation in phase timing,
  the targeted 14-test suite and two-rank Gloo/loader checks passed again.
- All 53 code cells of the current standalone notebook completed in native smoke
  mode on one RTX 3070 Ti Laptop GPU with four loader workers
  ([executed notebook](../validation_runs/executed_pq_speed_smoke.ipynb)).
- A synthetic CPU overlap benchmark (2048-square, 8 independent GT masks,
  256 disjoint predictions) gave median **4.9854 s original / 0.07079 s label
  counts**, about 70x for overlap calculation alone. Matrices were identical.
  This is not an epoch or Kaggle throughput claim
  ([measurements](../validation_runs/pq_overlap_benchmark.json)).
- The supplied Kaggle log shows 2.22 s/training batch, estimated 9.4 minutes of
  training per epoch, and 18.76 minutes preparing targets. Validation batch 20/36
  is not its start. See [log analysis](../validation_runs/pq_training_log_analysis.json).

The updated pipeline has not been benchmarked on dual T4s locally. The original
user settings were preserved except for the visible `PQ_EVERY=2` control; the
pre-update notebook is in `validation_runs/pre_pq_speed_notebook.ipynb`.

## T4 log fixes (2026-10-29)

- Reproduced the reported `(9,9,3,1)` versus `(9,1,3,1)` grayscale CUDA gradient
  mismatch and verified explicit channels-last transfer removes it. Regression
  coverage includes FP16, checkpointing, accumulation and a short final batch.
- CUDA suite with native probing: **34 passed, 1 skipped**
  ([XML](../validation_runs/logfix_gpu_tests.xml)); the skipped test requires two GPUs.
- CPU suite: **31 passed, 4 CUDA-only skipped**
  ([XML](../validation_runs/logfix_cpu_tests.xml)). Two-rank Gloo checks passed,
  including checkpointed channels-last convolution with gradient bucket views
  and no gradient-stride warning.
- Two-rank torchrun loopback rendezvous/all-reduce passed using a Windows-only
  legacy TCPStore test harness (the local wheel has no libuv). Production Linux
  NCCL transport was not exercised by that CPU check.
- Notebook source parity and all 52 code cells passed in a complete standalone
  native-resolution smoke run ([executed notebook](../validation_runs/executed_logfix_smoke.ipynb)).
  Hardware: one RTX 3070 Ti Laptop GPU, four loader workers for this test.
- The supplied dual-T4 log establishes batch two fits and batch four OOMs. Default
  probing now tests one and two. The pre-probe startup gap is not explained by
  the supplied excerpt; stage timings and explicit loopback rendezvous aid diagnosis.

User notebook code/settings outside runtime exports were preserved, with original
outputs saved in `validation_runs/pre_logfix_notebook.ipynb`. The updated notebook
clears old execution outputs. Dual-T4/NCCL execution remains unverified locally.

## Four-worker runtime update (2026-10-29)

- CPU suite: **29 passed, 3 CUDA-only skipped**
  ([XML](../validation_runs/parallel_cpu_tests.xml)).
- CUDA suite with the native 2048-square probe enabled: **31 passed, 1 skipped**
  ([XML](../validation_runs/parallel_gpu_tests.xml)). The skipped test requires
  two physical CUDA GPUs. Hardware was one RTX 3070 Ti Laptop GPU, 8 GiB.
- Two-rank CPU/Gloo smoke passed with four persistent loader workers sharing
  training/validation phases; accumulation matches the direct reference and
  validation counts each source once.
- Notebook structure/source parity passed: 30 sections, 19 modules, 101 cells.
- The complete standalone notebook ran successfully in smoke mode with four
  workers and one CUDA GPU ([executed notebook](../validation_runs/executed_parallel_smoke.ipynb)).
- Parallel/serial parity covers target arrays, statistics, PQ summaries, search
  trial order and tie handling, CSV bytes and journal resume. A guarded two-slot
  CPU model test confirms concurrent device scheduling and exclusive model use.

This update has not been executed on two T4s/NCCL, and no before/after throughput
claim is made. The earlier evidence below describes the previous runtime.

## Earlier validation (2026-10-19)

Verified 2026-10-19T13:57:18.268077+09:00 (Asia/Tokyo). Machine-readable evidence:
[`validation_summary.json`](../validation_runs/validation_summary.json).

| Check | Actual result |
|---|---|
| CPU synthetic/unit/integration suite | 26 passed, 2 CUDA-only skipped, 0 failures/errors |
| CUDA suite with native probe enabled | 28 passed, 0 skipped, 0 failures/errors |
| Two-rank CPU/Gloo DDP smoke | Passed: accumulation agrees with direct reference; unequal validation rank lengths count all records exactly once |
| Notebook structure/source parity | 30 ordered sections, 18 exported modules, 98 cells; generated modules match repository ASTs |
| Complete standalone notebook | All 50 code cells executed successfully with no cell errors |
| Full local annotation integrity | 8,199 annotations decoded and validated across 1,154 annotator records |
| Original native image checks | 707 train sources, 180 test sources; 2048 × 2048 |
| Native GPU training smoke | FP16, full-image forward/backward, accumulated AdamW, validation, best/last checkpoints and logs succeeded |
| Native validation/postprocessing | Successful cached full-image inference, two postprocessing comparisons and saved review figures |

Hardware: **NVIDIA GeForce RTX 3070 Ti Laptop GPU**, 8.00 GiB VRAM.
The standalone smoke used ModifiedUNet width 8, microbatch one and native 2048².
Real-backward probe peak allocated/reserved:
**1.718 / 1.811 GiB**.
AdamW optimizer state was allocated and updated during probing. This is local
hardware evidence, not a two-T4 throughput/memory claim.

The full dataset has 296 multiply annotated sources and
5 annotation records with overlapping original instance masks.
No pixel-identical duplicate filenames were found in this local training copy.
One decoded instance has area below 20 pixels (minimum 9 pixels), so minimum-area
filtering is a validation parameter, not a ground-truth deletion rule. The default
saved split contains 1013 train and 141 validation
records; smoke execution uses only two native records from each partition.

The one-epoch smoke validation PQ is **0.0**. It is
an execution check and does not establish competitive quality. No final competition
`submission.csv` was generated from the smoke model. Synthetic integration tests
exercise CSV generation, zero detections, RLE round trips, interruption recovery,
schema and duplicate checks.

## Evidence files

- [Executed notebook](../validation_runs/executed_smoke.ipynb)
- [CPU test XML](../validation_runs/pytest_cpu.xml) and [GPU test XML](../validation_runs/pytest_gpu.xml)
- [Gloo DDP log](../validation_runs/ddp_smoke.log)
- [Dataset integrity/statistics](../validation_runs/dataset_integrity.json)
- [Native smoke runtime](../validation_runs/notebook_execution/filament_smoke/runtime_report.json)
- [Native smoke history](../validation_runs/notebook_execution/filament_smoke/logs/history.csv)
- [Postprocessing trials](../validation_runs/notebook_execution/filament_smoke/search/postprocessing_trials.csv)
- [Training plot](../validation_runs/notebook_execution/filament_smoke/figures/training.png)
- [Target inspection](../validation_runs/notebook_execution/filament_smoke/figures/targets.png)
- [Prediction review](../validation_runs/notebook_execution/filament_smoke/figures/prediction_review.png)

Warnings observed: pycocotools emits a NumPy `__array__(copy=...)` deprecation
warning in the installed dependency combination. It does not fail round-trip or
evaluator tests. Notebook kernel startup emits a local Jupyter transport warning.

## Not executed and remaining limitations

Full dataset training, Linux/NCCL dual-T4 execution, sustained Kaggle memory/time
budget validation and Kaggle submission acceptance were not performed. Gloo checks
DDP synchronization mathematics, not NCCL deployment. Unexpected training OOM
restart is implemented but was not forced on a live multi-GPU job. Native GPU
smoke/probes passed without requiring width/batch fallback.

Two pixel heads cannot fully represent overlapping instance memberships. Watershed
can fragment elongated objects; optional reconnection can merge distinct objects.
Default single-annotator validation underrepresents annotator disagreement, and
checkpoint/postprocessing selection on one holdout creates selection bias. See
[EXPERIMENTS.md](EXPERIMENTS.md) for recommended controlled comparisons. No improved
competition performance is claimed.
