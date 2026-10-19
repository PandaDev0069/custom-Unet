# Implementation report

The requested native-resolution instance pipeline is implemented in the standalone
30-section [notebook](../solar_filament_unet_pq.ipynb) and reusable repository
modules. [VALIDATION.md](VALIDATION.md) records actual execution, with machine-readable
evidence and artifact links. Full Kaggle training and competitive quality have not
been claimed.

## Created and modified files

Modified original files:

- `unet.py`: configurable saved ModifiedUNet/DoubleConv, two channels by default,
  GroupNorm/activation selection, full-block checkpointing and odd-shape alignment.
- `filament_pq_proxy_loss.py`: explicit foreground/separation/exclusion targets,
  corrected contact-only legacy boundaries, per-image FP32 reductions; original
  soft morphology and clDice retained.

Created primary artifact and runtime:

- `solar_filament_unet_pq.ipynb`
- `filament/__init__.py`, `config.py`, `io.py`, `data.py`, `targets.py`
- `filament/model.py`, `experimental.py`, `losses.py`, `training.py`
- `filament/evaluation.py`, `postprocess.py`, `inference.py`, `submission.py`
- `filament/visualization.py`, `workflow.py`, `cli.py`

Created tests and development tools:

- `tests/conftest.py`, `test_data_targets.py`, `test_model_loss_training.py`
- `tests/test_evaluation_postprocess_submission.py`, `test_integration.py`
- `tests/reference_official.py`, `ddp_smoke_worker.py`
- `tools/build_notebook.py`, `verify_notebook.py`
- `tools/execute_notebook_smoke.py`, `write_validation_report.py`
- `pytest.ini`, `requirements.txt`, `.gitignore`

Created documentation:

- `README.md`
- `docs/PIPELINE.md`, `MODEL_ARCHITECTURE.md`, `LOSS_FUNCTIONS.md`
- `docs/TRAINING.md`, `EVALUATION.md`, `POSTPROCESSING.md`
- `docs/FUNCTION_REFERENCE.md`, `EXPERIMENTS.md`
- `docs/VALIDATION.md`, `IMPLEMENTATION_REPORT.md`

Successful local execution produced actual checkpoints, epoch logs, target caches,
training/inspection/error figures, validation probability maps, postprocessing
trial tables and reports under `validation_runs/`. These are smoke artifacts, not
a full trained competition model. Original reference notebooks, `memo.md`, `task.md`
and `download.svg` are left intact. Temporary notebook extraction copies used for
inspection were removed; the exact evaluator fixture remains in tests.

## Components reused and adapted

The saved `ModifiedUNet`, `DoubleConv`, max-pool encoder, transpose-convolution
decoder, skip connections, `_match_and_concat`, and raw-logit output remain the
primary architecture. The supplied `FilamentPQProxyLoss`, `soft_erode`, `soft_open`,
`soft_skeleton`, and `soft_cldice_loss` are reused/refined.

Infrastructure adapted from the full-image notebook includes `find_dataset_root`,
`make_records`, `FullImageDataset`, `probe_batch_sizes`, `run_epoch`,
`predict_full_image`, `component_map`, `mask_to_rle` and submission creation,
together with independent torchrun workers, AMP, accumulation, disk caching,
GroupNorm, channels-last, real-backward memory measurement and source-level splits.
The deeper `FullImageUNet`/global-context implementation remains explicitly opt-in.
Source audit details are in [PIPELINE.md](PIPELINE.md).

## Replacements and reasons

| Prior behavior | Replacement and reason |
|---|---|
| `polygons_to_mask` merged all labels and rounded polygons with OpenCV | Decode every COCO annotation independently through pycocotools, retaining membership; union is only a foreground target. |
| Single integer ID overwrote overlaps | -1 ambiguous IDs plus preserved original annotation masks/IDs; separation loss excludes overlap neighborhoods. |
| Exterior and inter-instance boundaries mixed together | Distinct-instance proximity target; isolated thin structures are not labeled as separation. |
| Boundary subtraction could erase narrow filaments | Marker extraction plus full-foreground refill and component fallback markers. |
| GT instances recovered as union connected components | Evaluate original independent annotation masks, including multipart and overlapping annotations. |
| Generic one-to-one PQ or foreground proxy selection | Exact organizer all-pairs IoU>0.5 semantics, global annotator-record aggregation, checkpoint selection by measured validation PQ. |
| Incomplete checkpoints | Model/optimizer/scheduler/scaler/epoch/history/best/early-stop/configuration/split and per-rank RNG state, written atomically. |
| Ad hoc threshold loops and final CSV | Provenance-bound validation cache/search, atomic per-source inference journals, complete CSV and source-accounting validation. |

## Checks actually performed

The final CUDA suite passed **28 tests**, including a native 2048² backward/AdamW
probe and AMP path. CPU testing passed **26 tests**, skipping the two CUDA-only
checks. Tests cover annotation formats, identity/overlap/contact/empty targets,
disconnected membership, split leakage and persistence, even/odd U-Net outputs,
loss gradients, exact incomplete accumulation, checkpoint/RNG restoration,
persistent-worker epoch state, optional experimental model, official evaluator
parity/grouping, all extraction methods, cache invalidation, RLE/schema and
interrupted inference. A separate two-rank CPU/Gloo smoke confirmed DDP accumulation
against a direct reference and disjoint unequal validation rank lengths.

All 8,199 local original annotations were decoded and checked across 1,154
annotator records and 707 training sources. Five records contain overlapping
instances. Native image headers for all 180 test sources were also checked. The
notebook exporter verifies 30 ordered sections, 18 source modules and 98 cells;
all 50 code cells were executed in an isolated standalone smoke run. Target,
training and prediction figure layouts were visually reviewed.

Detailed numbers, current source hashes and actual memory readings are in the
generated [validation report](VALIDATION.md), not estimated from the architecture.

## Unexecuted paths and recommended experiments

Full-dataset optimization, dual-T4 Linux/NCCL execution, sustained Kaggle resource
budget verification and Kaggle CSV acceptance were not performed. Whole-job OOM
recovery is implemented; a live multi-GPU OOM was not forced. The native smoke
checkpoint has PQ 0.0 after one epoch over two training records and is deliberately
refused for final submissions. It establishes execution only.

The two-head representation cannot express overlapping predicted memberships and
cannot infer arbitrary disconnected-instance identity. Watershed can fragment
filaments; continuity refinement can overmerge. The default validation policy
underrepresents annotator disagreement, and shared checkpoint/postprocessing
selection can bias the holdout score. Recommended experiments are width after
real memory probing, separation radius/positive weight, seed spacing, grouped
multi-annotator validation, and clDice only after profiling. See
[EXPERIMENTS.md](EXPERIMENTS.md) for controls and measurements.
