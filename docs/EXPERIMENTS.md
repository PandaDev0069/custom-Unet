# Reproducing and comparing experiments

Use a separate output directory for each experiment. Archive requested/effective
configuration, split manifest, annotation/content fingerprints, package/hardware
report, checkpoints, loss/PQ history, selected postprocessing and validation details
together. A checkpoint alone cannot reproduce the instance predictions without
normalization and postprocessing parameters.

## Baseline sequence

1. Run all synthetic tests and the native smoke command with the intended GPU
   environment. Inspect target overlays before allocating a long training run.
2. Set notebook `MODE='full'`; retain two output channels, clDice zero, native
   2048², GroupNorm, checkpointing and width 16 initially. Real probes may select a
   smaller width; use the effective configuration when reporting results.
3. Train using the saved source split. Review both loss curves and instance PQ.
   Record wall time and GPU/CPU throughput rather than assuming dual GPUs double
   speed on a one-core host.
4. Cache validation predictions once from `best.pt`; compare component, boundary
   and watershed extraction using validation PQ. Save trials and selected JSON.
5. Inspect false positives, misses, fragmented and merged examples at native
   pixels. An improved foreground Dice alone does not demonstrate improved PQ.
6. Freeze model, preprocessing and postprocessing, then infer test images and
   validate the counts-only RLE submission. Resume the same journals if interrupted.

The notebook executes these stages in the requested 30-section order. CLI
`train`, `evaluate`, `infer` reproduces the same stages using repository modules.
Use a CUDA-enabled Python executable throughout; the default host Python on this
machine is CPU-only, while the project's `.venv` supports CUDA.

## Controlled comparisons

| Experiment | Fixed controls | What to measure |
|---|---|---|
| Width 8/16/32 | Split, losses, augmentation and search grid | Global PQ, peak/reserved VRAM, training time and effective batch. |
| Separation radius 1/2/3 | Architecture, split and optimizer | Contact recall, over-segmentation, narrow-structure preservation and PQ. |
| Separation positive weight | Same cached target policy | PQ and probability threshold sensitivity, not just BCE. |
| Components/boundary/watershed | Exactly the same checkpoint/cache | PQ, TP/FP/FN, matched IoU, degree relations and CPU time. |
| Conservative continuity | Same seeds and thresholds | Recovered disconnected memberships versus overmerged predictions. |
| clDice > 0 | Full real-backward memory probe first | Topology changes, instance PQ, additional VRAM/time. |
| Grouped multi-annotator validation | New frozen split/output directory | Generalization under disagreement; global per-record aggregation. |
| Source date grouping | New split protocol | Sensitivity to temporal correlations not prevented by exact pixel grouping. |

Changing target radius produces a new cache fingerprint. Changing the validation
inventory requires a new manifest/output directory. Resume is for interrupted
training under consistent semantics, not for quietly changing an experiment's
targets or loss. Explicitly start a new experiment for such changes.

The same holdout selects both checkpoint and postprocessing, which biases its
reported score upward. Reserve an untouched grouped holdout or perform nested
cross-validation before claiming a performance improvement. Never tune using test
ground-truth metadata or select by Kaggle public-leaderboard feedback.

## Development results and limitations

Actual local validation is recorded in [VALIDATION.md](VALIDATION.md) and
`validation_runs/`. Native smoke runs demonstrate execution only. Full dataset
training, Kaggle submission acceptance and dual-T4/NCCL throughput have not been
measured in this implementation session. No competitive-performance claim is made.

The two-head representation predicts one non-overlapping label assignment at
inference. It cannot fully encode multiple instance memberships at crossings.
Continuity union can associate disconnected pieces but cannot infer arbitrary
annotation identity. If these cases dominate validation error, a future model
may need embeddings, queries or seed-conditioned instance refinement; evaluate
that as a separate architecture experiment while keeping original GT/evaluator.
