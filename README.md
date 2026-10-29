# Native-resolution solar filament instances

Start with [solar_filament_unet_pq.ipynb](solar_filament_unet_pq.ipynb). It is a
self-contained Kaggle notebook: attach the competition dataset, select a CUDA
accelerator, and run top to bottom. No custom module uploads are needed. It
exports readable Python modules with `%%writefile`, then launches an independent
training worker. The notebook defaults to `MODE = 'full'`, four CPU workers, and
two T4 GPUs. Select **GPU T4 x2** in Kaggle, restart the kernel, and run all cells.
Set `MODE = 'smoke'` to run two native images per partition for one epoch.

The four loader workers are split two per GPU and reused for training/validation.
Preparation, statistics, postprocessing search, and submission encoding use bounded
four-thread pools. Validation and test inference keep one model on each GPU and
process full images concurrently. FP16, pinned/prefetched loading, channels-last,
and cuDNN autotuning are enabled; real-backward probes select microbatch 1, 2, or 4
within each GPU's memory limit. The two 15 GiB memories remain separate.

The primary model is the saved four-pool `ModifiedUNet`, extended to foreground
and separation logits. Original independent COCO annotation masks remain intact,
including overlaps and disconnected parts. Ground truth is never reconstructed
from connected components of a union mask. Separation targets describe contacts
between distinct annotations; overlap neighborhoods are explicitly excluded from
separation supervision. The objective is a differentiable surrogate for instance
quality, not official PQ itself.

The evaluator follows the supplied organizer notebook exactly: every pair with
IoU **strictly greater than 0.5** counts, unmatched rows/columns produce FN/FP, and
counts are aggregated globally across annotator-image records. Overlapping GT can
match the same prediction more than once under these semantics. See
[EVALUATION.md](docs/EVALUATION.md).

## Local use

Use a CUDA-enabled PyTorch environment (Python 3.12 recommended). Keep Kaggle's
GPU PyTorch installation; the notebook installs only missing auxiliary packages.
`requirements.txt` records the tested dependency versions. Do not replace CUDA
PyTorch with a CPU build when setting up locally.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python tests/ddp_smoke_worker.py
python -m filament.cli smoke --gpus 1 --output validation_runs/local_smoke
```

The MAGFiLO root can be discovered from an ancestor directory, or set explicitly
in a JSON configuration. The expected structure is:

```text
MAGFilo_1.0_Kaggle_2026/
  train/train_images/*.jpeg
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/*.jpeg
```

For a full run, export and edit the central configuration:

```python
from filament.config import Config
from filament.io import atomic_json
cfg = Config(data_root='/kaggle/input/.../MAGFilo_1.0_Kaggle_2026',
             output_dir='/kaggle/working/filament_full')
atomic_json('config.json', cfg.to_dict())
```

```bash
python -m filament.cli train --config config.json --gpus 2
python -m filament.cli evaluate --config config.json
python -m filament.cli infer --config config.json
```

`--gpus 2` requires Linux/NCCL. One GPU works on Windows. `prepare` indexes,
validates, splits and caches without starting training. `evaluate` caches
validation maps, compares postprocessing settings and generates review plots.
`infer` requires the selected postprocessing JSON and checks its checkpoint hash.
Smoke checkpoints are refused for final test submissions.

## Artifacts

Each experiment keeps `config.json`, `effective_config.json`, a split manifest,
data reports, fingerprinted targets, `checkpoints/{best,last}.pt`, epoch CSV/logs,
validation reports, prediction caches, figures, postprocessing trials and winning
JSON. Successful test inference creates `submission.csv` plus a manifest with all
source counts, including zero detections. Per-source journals allow resuming an
interruption. No final competition CSV is fabricated during development.

Local verification artifacts are under `validation_runs/`. Their smoke scores are
execution checks, not estimates of competition performance. The full training run
and dual-T4 NCCL execution have not been claimed. See
[VALIDATION.md](docs/VALIDATION.md) for actual evidence and limitations.

## Documentation

| Guide | Contents |
|---|---|
| [PIPELINE](docs/PIPELINE.md) | Data, annotation policy, source audit, complete workflow |
| [MODEL_ARCHITECTURE](docs/MODEL_ARCHITECTURE.md) | Saved U-Net, tensor shapes, modifications |
| [LOSS_FUNCTIONS](docs/LOSS_FUNCTIONS.md) | Mathematics, exclusions, AMP, extending losses |
| [TRAINING](docs/TRAINING.md) | Probes, DDP, accumulation, checkpoint recovery, resource budget |
| [EVALUATION](docs/EVALUATION.md) | Official scoring, reference tests and analysis |
| [POSTPROCESSING](docs/POSTPROCESSING.md) | Algorithms, validation-only parameter search |
| [FUNCTION_REFERENCE](docs/FUNCTION_REFERENCE.md) | Public functions, inputs and outputs |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | Reproducible experiments and recommended comparisons |

To edit the runtime, change repository modules, rebuild the notebook with
`python tools/build_notebook.py`, and run `python tools/verify_notebook.py` from an
environment containing IPython/nbformat. `tools/execute_notebook_smoke.py` executes
the entire standalone notebook in an isolated directory with the current Python.

Competition sources: [challenge](https://www.kaggle.com/competitions/filament-segmentation-2026/),
[submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission),
and [organizer self-evaluation](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
The local supplied evaluator is the implemented scoring reference; its strict
pair threshold takes precedence over generic PQ implementations.
