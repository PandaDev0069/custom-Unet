"""Assemble a readable, self-contained Kaggle notebook from canonical modules.

Source cells use %%writefile, including append cells at AST statement boundaries;
there are no encoded archives or giant Python source string literals to inspect.
"""
import ast
import hashlib
import json
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build():
    nb=nbf.v4.new_notebook()
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md("""# 1. Solar filament instance segmentation at native resolution

This notebook trains the supplied **ModifiedUNet** on complete 2048 × 2048
grayscale H-alpha images. Two raw-logit channels predict foreground and
inter-instance separation. Individual COCO annotations, including overlaps and
disconnected regions, remain the ground truth for instance evaluation.

Attach the official competition dataset in Kaggle and enable two T4 GPUs (or one
CUDA GPU). No extra source uploads are required: readable cells below export the
entire runtime. Defaults use **4 CPU workers and 2 T4 GPUs** for full training.
Set `MODE = 'smoke'` in section 5 to run two native images per partition for one
epoch. Smoke checkpoints never produce a final test submission.

Sources: [competition](https://www.kaggle.com/competitions/filament-segmentation-2026),
[organizer evaluator](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook),
and the supplied local notebooks/model/loss. See repository docs for the source audit.
""")
    md(r"""## 2. Evaluation and PQ mathematics

For independent masks $G_i,P_j$, $I_{ij}=|G_i\cap P_j|/|G_i\cup P_j|$.
The supplied organizer notebook marks **every pair with $I_{ij}>0.5$** as a true
positive. It does not perform Hungarian assignment. FP counts columns with no
qualifying match; FN counts unmatched rows. Aggregate over all annotator-image
records, then compute

$$PQ={\sum_{(i,j):I_{ij}>0.5} I_{ij}\over TP+\tfrac12 FP+\tfrac12 FN}.$$

The denominator-zero convention is 0. Predictions of a source image are scored
separately against each annotator. This matters when annotations overlap.
Dice/IoU distributions include all **positive-overlap pairs**; fragmentation and
merging use degree at IoU > 0. Validation uses original annotation masks, never
connected components of the foreground union. Validation selection is a global
instance PQ; loss is only a differentiable surrogate and training diagnostic.
""")
    md("""## 3. Pipeline architecture

```text
COCO annotations ──> independent instance masks ─┬─> foreground union
                                               ├─> separation + overlap exclusion
                                               └─> original GT for PQ
Native image ─> normalize ─> augment ─> ModifiedUNet ─> [foreground, separation]
                            train with AMP/DDP          │
                     best validation PQ checkpoint <───┤
                                    validation cache ─> postprocess search
                              test full-image forward ─> instances ─> COCO RLE CSV
```

The default split keeps multiply annotated sources in training, as in the prior
full-image notebook. All records of a filename and all pixel-identical duplicates
stay together. No test annotations are read. A two-head pixel representation
cannot fully recover crossing/overlapping identities or arbitrary disconnected
membership; these remain explicit modeling limitations.
""")
    md("""## 4. Dependencies and runtime checks

Kaggle supplies its CUDA-enabled PyTorch. Keep that build. The following cell
installs missing auxiliary packages, then exports the reusable source modules as
ordinary Python files. Each source cell is readable and can be edited directly.
The repository `requirements.txt` records a tested auxiliary package set.
""")
    code("""import importlib.util, os, sys, subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Each CPU worker gets one native-library thread; avoid nested thread pools.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
missing = [package for module, package in [('pycocotools','pycocotools==2.0.11'),
    ('scipy','scipy==1.18.1'), ('skimage','scikit-image==0.26.0'),
    ('cv2','opencv-python-headless==5.0.0.93'), ('pandas','pandas==3.0.6'),
    ('PIL','Pillow==12.3.0'), ('matplotlib','matplotlib==3.11.2')]
    if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
SOURCE_DIR = Path.cwd() / 'filament_runtime'
(SOURCE_DIR / 'filament').mkdir(parents=True, exist_ok=True)
print('Exporting self-contained runtime to', SOURCE_DIR)
""")
    files=[ROOT/'unet.py',ROOT/'filament_pq_proxy_loss.py',*sorted((ROOT/'filament').glob('*.py'))]
    for file in files:
        source=file.read_text(encoding='utf8')
        relative=file.relative_to(ROOT).as_posix()
        md(f"Runtime source: `{relative}`")
        lines=source.splitlines(keepends=True)
        tree=ast.parse(source)
        chunks=[]; start=0
        for node in tree.body:
            end=node.end_lineno
            if end-start>160 and node.lineno-1>start:
                chunks.append(''.join(lines[start:node.lineno-1])); start=node.lineno-1
        chunks.append(''.join(lines[start:]))
        for i,chunk in enumerate(chunks):
            code(f"%%writefile {'-a ' if i else ''}filament_runtime/{relative}\n{chunk.rstrip()}\n")
    code("""sys.path.insert(0, str(SOURCE_DIR.resolve()))
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
print('Python:', sys.version, '| PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
for i in range(torch.cuda.device_count()):
    gpu = torch.cuda.get_device_properties(i)
    print(i, gpu.name, f'{gpu.total_memory / 2**30:.2f} GiB')
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle GPU accelerator or use a CUDA-enabled local environment.')
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
""")
    md("""## 5. Imports and configuration

All major choices live in `Config`. Width 16, GroupNorm, non-reentrant activation
checkpointing, FP16, microbatch probing, and accumulation are conservative starting
points for T4s. Microbatches 1 and 2 are probed within each 15 GiB device's
memory limit; the supplied dual-T4 log confirms 2 fits and 4 OOMs. Width 8 is an
automatic pre-training fallback; image resolution
never changes. Four workers are a **total** CPU budget: two loader processes per
T4 during training/validation; four CPU threads for preparation, search and RLE.
Both GPUs also run validation/test inference. Native-library threads are capped
at one per worker. Plotting, reductions and final file assembly remain ordered.
""")
    code("""from filament.config import Config
from filament.io import atomic_json, file_hash
from filament.data import *
from filament.targets import *
from filament.model import build_model
from filament.losses import FilamentPQProxyLoss
from filament.workflow import prepare_experiment, launch_training
from filament.training import load_checkpoint, seed_everything
from filament.inference import *
from filament.evaluation import *
from filament.postprocess import extract_instances
from filament.submission import *
from filament.visualization import *

MODE = 'full'  # 'smoke' or 'full'
CPU_WORKERS = 4
REQUESTED_GPUS = 2
PQ_EVERY = 2  # Validation loss every epoch; instance PQ on first/every 2nd/final epoch.
DATA_ROOT_OVERRIDE = ''
RUN_TRAINING = True
RUN_TEST_INFERENCE = MODE == 'full'
ON_KAGGLE = os.name != 'nt' and Path('/kaggle/working').exists() and Path('/kaggle/input').exists()
OUTPUT_DIR = (Path('/kaggle/working') if ON_KAGGLE else Path.cwd()) / f'filament_{MODE}'
cfg = Config(data_root=DATA_ROOT_OVERRIDE, output_dir=str(OUTPUT_DIR), smoke=MODE == 'smoke',
             workers=CPU_WORKERS, prefetch_factor=2, deterministic=False, pq_every=PQ_EVERY)
# deterministic=False enables cuDNN autotuning for fixed native image shapes.
# Set it True for deterministic kernels, at a potential throughput cost.
if MODE == 'smoke':
    cfg.epochs = 1
    cfg.model['init_features'] = 8
    cfg.microbatch = 1
# Resume a complete epoch: cfg.resume = str(OUTPUT_DIR / 'checkpoints/last.pt')
cfg.validate()
seed_everything(cfg.seed, cfg.deterministic)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(cfg.to_dict())
""")
    md("""## 6. Dataset discovery

Find `train/train_images`, the original MAGFiLO annotation JSON, and
`test/test_images`. Inspect native dimensions, image IDs, missing files and
pixel-identical duplicates. Test image pixels are used only during final inference.
""")
    code("""DATA_ROOT = find_dataset_root(cfg.data_root)
records, integrity = make_records(DATA_ROOT, workers=cfg.workers)
integrity['test_source_images'] = len(discover_test_images(DATA_ROOT, workers=cfg.workers))  # Headers/inventory only; no test tuning.
print({k:v for k,v in integrity.items() if k != 'multi_annotator_sources'})
print('Multi-annotator source count:', len(integrity['multi_annotator_sources']))
atomic_json(OUTPUT_DIR / 'data/integrity.json', integrity)
""")
    md("""## 7. Annotation inspection

One COCO image record identifies one annotator and source. Multiple polygons in
one annotation are decoded as one independent instance using pycocotools. Empty
annotation lists are valid images with zero instances; a malformed or empty
individual annotation raises an error and is never silently removed.
""")
    code("""example = records[0]
instances = load_image_instances(example)
print(example['image_id'], example['file_name'], 'independent instances:', len(instances))
print('Native dimensions:', example['height'], example['width'])
del instances
""")
    md("""## 8. Dataset statistics

Statistics use a bounded four-worker pool. Smoke mode inspects a small sample; full
mode validates every original annotation. Area histograms, foreground fraction,
overlap pixels and small-instance counts are saved without retaining dense masks.
""")
    code("""stats_records = records[:8] if cfg.smoke else records
statistics, statistics_rows = summarize_dataset(stats_records, workers=cfg.workers)
atomic_json(OUTPUT_DIR / 'data/statistics.json', statistics)
pd.DataFrame(statistics_rows).to_csv(OUTPUT_DIR / 'data/statistics_per_record.csv', index=False)
display(pd.DataFrame(statistics_rows).head())
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist([r['instances'] for r in statistics_rows], bins=20); axes[0].set_title('Instances per annotator record')
axes[1].hist(statistics['instance_areas'], bins=40); axes[1].set(xscale='log', title='Instance areas (native pixels)')
axes[2].hist([r['foreground_ratio'] for r in statistics_rows], bins=20); axes[2].set_title('Foreground fraction')
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'data/statistics.png', dpi=180); plt.show()
""")
    md("""## 9. Train/validation splitting

Multi-annotator source files stay in training by default. Validation size targets
20% of unique source filenames and is drawn only from eligible single-annotator
groups. Duplicate pixel groups remain intact. The manifest binds image identities
and is reused rather than resampled. This policy facilitates baseline comparison
but validation does not represent annotator disagreement; grouped cross-validation
including multiple annotators is a recommended subsequent experiment.
""")
    code("""split_path = OUTPUT_DIR / 'split_manifest.json'
split = load_split_manifest(split_path, records) if split_path.exists() else create_train_val_split(
    records, cfg.val_fraction, cfg.seed, cfg.multi_annotator_train_only)
save_split_manifest(split_path, split)
by_id = {r['image_id']:r for r in records}
verify_no_leakage([by_id[i] for i in split['train']], [by_id[i] for i in split['val']])
print('Training records:', len(split['train']), '| Validation records:', len(split['val']))
""")
    md("""## 10. Target generation and validation

Foreground is the union of all independent masks. IDs are 0 for background,
positive for unambiguous annotation membership, and -1 for overlap. Separation is
foreground near **two different annotations**, not the contour of one filament.
Overlap and a radius neighborhood are excluded from separation BCE. Foreground
overlap remains supervised; originals are retained in compact COCO JSON for PQ.
Preparation writes compressed atomic per-record arrays with content fingerprints.
```text
Annotation polygons / RLE → one independent mask per annotation
                       ├→ union → foreground
                       ├→ distinct-mask proximity → separation
                       ├→ overlapping coverage → exclusion mask / ID = -1
                       └→ originals retained → instance PQ
```
""")
    code("""runtime = prepare_experiment(cfg, statistics=False, indexed=(records, integrity))
train_records, val_records = runtime['train_records'], runtime['val_records']
figure = plot_dataset_inspection(val_records[0], OUTPUT_DIR / 'figures/targets.png')
plt.show(); plt.close(figure)
print('Prepared:', len(train_records), len(val_records))
""")
    md("""## 11. Full-resolution preprocessing

Read native uint8 grayscale JPEG; divide by 255 to float32. Optional `robust`
mode clips/normalizes the 1st–99th intensity percentiles consistently in train,
validation and test. No resizing or tiling occurs. Skip alignment pads only a
rounding difference when odd test dimensions are used in synthetic checks.
""")
    code("""native_image = read_grayscale(val_records[0]['image_path'], (2048, 2048))
normalized = normalize_image(native_image, cfg.normalization)
assert normalized.shape == (2048, 2048) and normalized.dtype == np.float32
print(normalized.shape, float(normalized.min()), float(normalized.max()))
del native_image, normalized
""")
    md("""## 12. Augmentation

Flips and 90-degree rotations are shared across every target map. Mild contrast,
brightness, gamma and optional noise affect only the input. The RNG is indexed by
seed, epoch and record; validation/test are deterministic. Native morphology is
preserved without interpolation or arbitrary-angle resampling.
""")
    code("""preview_dataset = FullImageDataset(train_records, True, cfg.seed, cfg.normalization, cfg.augment)
preview_dataset.set_epoch(1)
sample = preview_dataset[0]
print({k:tuple(v.shape) for k,v in sample.items() if hasattr(v, 'shape')})
del sample, preview_dataset
""")
    md("""## 13. Dataset and DataLoader

Lazy loading returns input and three B1HW supervision maps, not an all-dataset
instance stack. Training DDP uses `DistributedSampler.set_epoch`; validation
partitions have no padding duplicates. Pinned memory, configurable prefetch and
persistent workers are enabled only when workers > 0. Shared epoch state keeps
persistent-worker augmentations correct. One persistent pool serves both phases,
so four workers are not doubled during validation. Training sampler padding can repeat a
small number of records to keep rank lengths equal.
""")
    code("""validation_dataset = FullImageDataset(val_records, normalization=cfg.normalization)
print('Input/targets are loaded on demand:', len(validation_dataset), 'validation records')
""")
    md("""## 14. Custom U-Net architecture

```text
1 × 2048² → f × 2048² ───────────────────────────────┐
             ↓ maxpool                             │ skip
           2f × 1024² ───────────────────────┐       │
             ↓                              │       │
           4f × 512² ────────────────┐       │       │
             ↓                      │       │       │
           8f × 256² ───────┐       │       │       │
             ↓              │       │       │       │
          16f × 128² → transpose conv + concat → DoubleConv (reverse levels)
                                                     ↓
                                              1×1 conv → 2 × 2048²
                                                 foreground / separation
```

This preserves the saved `ModifiedUNet` architecture, `_match_and_concat`, and
raw logits. GroupNorm supports batch one. BatchNorm is available only with
checkpointing disabled to avoid recomputation updating running statistics twice.
The previous deeper `FullImageUNet` remains opt-in via `name='full_image'`, with
`base_channels`, `depth`, `max_channels`, and `checkpoint_blocks` parameters.
""")
    code("""probe_model = build_model(dict(name='modified', init_features=2, checkpoint_blocks=False))
with torch.inference_mode():
    for height, width in [(32, 48), (33, 35)]:
        assert probe_model(torch.zeros(1, 1, height, width)).shape == (1, 2, height, width)
print('Even/odd two-channel shape checks passed')
del probe_model
""")
    md(r"""## 15. Loss functions and mathematics

$$L=w_b\,BCEWithLogits(F,Y)+w_d\left(1-{2\sum\sigma(F)Y+1\over\sum\sigma(F)+\sum Y+1}\right)
+w_s\,BCE_{posweight}(S,T_s;V)+w_c\,clDice.$$

$V$ excludes ambiguous overlap neighborhoods. Separation BCE is averaged over
valid pixels **per image**, then over images. Foreground BCE and soft Dice retain
all union pixels. FP32 arithmetic is enforced even under AMP. Each term is logged.
The supplied soft erosion/opening/skeleton/clDice functions are reused. clDice is
disabled by default because native-resolution skeleton graphs consume substantial
VRAM. This objective is a surrogate, not a differentiable official PQ.
```text
Foreground logits → BCE + soft Dice (+ optional clDice) ─┐
Separation logits + target + valid mask → weighted BCE ──┤
                         configurable weighted sum ←───┘
                                     ↓
                            AMP-scaled backward
```
""")
    code("""small_model = build_model(dict(name='modified', init_features=2))
logits = small_model(torch.randn(1, 1, 33, 35))
target = {'foreground':torch.zeros(1,1,33,35), 'separation':torch.zeros(1,1,33,35),
          'separation_valid':torch.ones(1,1,33,35)}
terms = FilamentPQProxyLoss(**cfg.loss)(logits, target)
terms['loss'].backward()
assert torch.isfinite(terms['loss'])
print({k:float(v.detach()) for k,v in terms.items()})
del small_model, logits, target, terms
""")
    md("""## 16. GPU setup and memory probing

Each worker probes complete native images with the configured loss, accumulated
backward, AMP and **AdamW state allocation**. Allocated/reserved/peak memory and
DDP headroom are measured. Probes run before DDP wrapping, so an OOM can be handled
without trapping other ranks in a collective. Weights and RNG are restored.
All GPUs must pass a common microbatch. Width fallback is explicit in runtime logs.

```text
Notebook → torchrun → rank 0/T4 → full-image batch → gradients ─┐
                   → rank 1/T4 → full-image batch → gradients ─┤ all-reduce
                                                        AdamW update on each rank
```

VRAM is not pooled across GPUs. Unexpected training OOM triggers a whole-job
restart from the last complete epoch using a smaller microbatch. Width changes
are permitted only before a checkpoint exists. No automatic spatial reduction.
""")
    code("""NUM_GPUS = min(REQUESTED_GPUS, torch.cuda.device_count())
if os.name == 'nt': NUM_GPUS = 1
print('Worker GPU count:', NUM_GPUS)
if NUM_GPUS < REQUESTED_GPUS:
    print('Using available/supported GPUs; select GPU T4 x2 in Kaggle settings for both T4s.')
print('Total CPU workers:', cfg.workers, '| workers per GPU:',
      [cfg.workers // NUM_GPUS + int(rank < cfg.workers % NUM_GPUS) for rank in range(NUM_GPUS)])
print('Nominal effective batch = selected microbatch ×', NUM_GPUS, '×', cfg.accumulation_steps)
""")
    md("""## 17. Training configuration

AdamW, clipping, AMP scaling, plateau/cosine/no scheduler, exact short-window
accumulation, early stopping by PQ evaluations, and best/last atomic checkpoints
are configured centrally. Resume requires the same split, loss, preprocessing,
augmentation, accumulation, scheduler and GPU count. The effective model width
and batch are persisted with checkpoints. Epoch RNG state is stored per rank.
""")
    code("""atomic_json(OUTPUT_DIR / 'requested_config.json', cfg.to_dict())
print('Epochs:', cfg.epochs, '| PQ frequency:', cfg.pq_every, '| clDice:', cfg.loss['cldice_weight'])
""")
    md("""## 18. Training execution

An independent generated worker process avoids notebook DDP spawning issues.
Logs stream to this cell and disk. Interrupting training leaves the last complete
epoch intact; set `cfg.resume` in section 5 and rerun preparation/training.
Rank-labelled startup timings identify import, CUDA, NCCL, probe and loader waits.
Single-host torchrun uses an explicit loopback rendezvous. Smoke mode verifies
execution, not model quality. Each epoch logs separate training/validation wall
times and maximum per-rank CPU postprocessing/PQ-scoring times. Validation loss
runs every epoch; `PQ_EVERY=2` evaluates instance PQ on the first, every second and
final epoch. Set it to 1 for every-epoch checkpoint selection. Early-stopping
patience counts PQ evaluations. All strict IoU>0.5 pairs still count, including
overlapping GT; disjoint prediction IDs accelerate overlap counting on the CPU.
""")
    code("""if RUN_TRAINING:
    # Avoid retaining parent-process CUDA models while workers train.
    import gc
    gc.collect(); torch.cuda.empty_cache()
    launch_training(runtime, NUM_GPUS)
else:
    print('Using existing checkpoint; ensure OUTPUT_DIR is the intended experiment.')
""")
    md("""## 19. Training visualization

Loss terms, validation PQ, learning rate, epoch/total duration and per-rank GPU
statistics are persisted. Utilization is sampled where NVML is available; it is
not an epoch-average measurement. The training graph distinguishes diagnostics
from the instance objective used to select the best checkpoint.
""")
    code("""history = pd.read_csv(OUTPUT_DIR / 'logs/history.csv')
display(history.tail())
fig = plot_training(OUTPUT_DIR / 'logs/history.csv', OUTPUT_DIR / 'figures')
plt.show(); plt.close(fig)
""")
    md("""## 20. Checkpoint loading

Use the checkpoint's **effective** architecture and preprocessing after probing,
not the requested width. Only load trusted locally generated PyTorch checkpoints.
The best checkpoint is selected by strict improvement in global validation PQ.
""")
    code("""CHECKPOINT = OUTPUT_DIR / 'checkpoints/best.pt'
state = load_checkpoint(CHECKPOINT)
effective_cfg = Config(**state['config'])
DEVICE = torch.device('cuda:0')
model = build_model(effective_cfg.model).to(DEVICE)
model.load_state_dict(state['model']); model.eval()
if effective_cfg.channels_last: model.to(memory_format=torch.channels_last)
predictor = make_predictor(model, effective_cfg, DEVICE, NUM_GPUS)
print('Selected epoch:', state['epoch'], '| validation PQ:', state['best_metrics'].get('pq'))
del state
PROVENANCE = {'checkpoint_sha256':file_hash(CHECKPOINT), 'normalization':effective_cfg.normalization,
              'split_fingerprint':runtime['split_fingerprint']}
""")
    md("""## 21. Full-resolution validation inference

Run one forward pass per unique validation filename across both GPUs and cache
two float16 maps. Four bounded CPU jobs overlap image reads, GPU work and writes.
Cache keys include checkpoint, normalization, inference options and source file
hash. Search reads maps with memory mapping and never repeats model inference.
Inference during training uses FP32 probabilities; search uses float16 caches,
which can shift pixels extremely close to a threshold. This precision tradeoff is
recorded and should be checked in a final experiment if consequential.
""")
    code("""prediction_manifest = cache_validation_predictions(model, val_records, OUTPUT_DIR / 'validation_predictions',
    DEVICE, PROVENANCE, workers=cfg.workers, predictor=predictor,
    amp=effective_cfg.amp, channels_last=effective_cfg.channels_last,
    normalization=effective_cfg.normalization)
print('Cached source predictions:', len(prediction_manifest['entries']))
""")
    md("""## 22. Instance extraction and postprocessing

Compare components, separation-aware connected cores with geodesic refill, and
marker watershed using distance peaks and a separation elevation term. Every
foreground component gets a fallback marker even if entirely classified as a
separator. Small-instance filtering happens after refill. Optional continuity
refinement joins only short, mutually nearest PCA-aligned, probability-supported
gaps; it is disabled by default and may still merge distinct filaments.

```text
Foreground threshold → optional closing → seed extraction ← separation
 → component / geodesic refill / marker watershed → min area → IDs → independent masks
```
""")
    code("""baseline_summary, baseline_details = evaluate_cached_predictions(val_records, prediction_manifest,
    effective_cfg.postprocess, workers=cfg.workers)
print(baseline_summary)
""")
    md("""## 23. Official PQ evaluation

Global aggregation retains every annotator record and every independent mask.
The repository tests compare matrices, grouping, strict threshold, empty cases,
overlapping annotations and final score with functions extracted unchanged from
the supplied self-evaluation notebook. The record API additionally includes empty
GT records that cannot be represented in an instance-only CSV.
```text
Each original annotator record + source predictions → GT × prediction overlaps
 → every IoU > 0.5 pair: TP / IoU sum; unmatched columns: FP; unmatched rows: FN
 → sum across annotator records → global PQ ratio
```
""")
    code("""synthetic = np.zeros((16,16), bool); synthetic[2:12,3:7] = True
assert evaluate_panoptic_quality([synthetic, synthetic], [synthetic])['tp'] == 2
assert evaluate_panoptic_quality([synthetic], [synthetic])['pq'] == 1
assert evaluate_panoptic_quality([], [])['pq'] == 0
atomic_json(OUTPUT_DIR / 'validation/baseline.json', {'summary':baseline_summary, 'records':baseline_details})
print('Global PQ:', baseline_summary['pq'])
""")
    md("""## 24. Dice, IoU and fragmentation analysis

Positive-pair distributions and IoU>0 degree counts reproduce the reference
analysis. Matched IoU is reported separately from the positive-pair mean. Tiny
incidental overlaps also contribute to fragmentation/merging degrees, so inspect
the spatial examples before diagnosing an error from a count alone.
""")
    code("""fig = plot_distributions(baseline_details, OUTPUT_DIR / 'figures/overlap_distributions.png')
plt.show(); plt.close(fig)
display(pd.DataFrame([{k:r[k] for k in ['file_name','pq','tp','fp','fn','fragmented_gt','merged_predictions','n_pred']}
    for r in baseline_details]).sort_values('pq').head())
""")
    md("""## 25. Validation-based postprocessing optimization

No retraining: reuse cached maps. First configuration wins ties. Save all trials
to CSV and the winning parameters/provenance to JSON. Four CPU workers evaluate
trials concurrently with bounded memory and deterministic tie order. This holdout
is used for model and threshold selection, so its score is selection-biased;
reserve another grouped holdout or use cross-validation for a final estimate.
""")
    code("""SEARCH_GRID = {'method':['components','boundary','watershed'],
    'foreground_threshold':[.4,.5,.6], 'boundary_threshold':[.4,.6], 'min_area':[10,20]}
if cfg.smoke:
    SEARCH_GRID = {'method':['components','boundary'], 'foreground_threshold':[.5]}
# Optional grid entries: seed_distance, seed_min_area, boundary_weight,
# closing_radius, reconnect_distance (only for method='continuity').
best_postprocessing, trials = search_postprocessing(val_records, prediction_manifest, SEARCH_GRID,
    OUTPUT_DIR / 'search', effective_cfg.postprocess, workers=cfg.workers)
POSTPROCESS = best_postprocessing['postprocess']
display(trials.sort_values('pq', ascending=False).head())
""")
    md("""## 26. Qualitative prediction review

Save full disks and native-pixel zooms for best/worst PQ, false positives, missed
filaments, fragmentation and merging when those error cases exist. Crops are
display-only. The probability maps always come from complete-image inference.
""")
    code("""selected_summary, selected_details = evaluate_cached_predictions(val_records, prediction_manifest, POSTPROCESS, workers=cfg.workers)
gallery = plot_error_gallery(val_records, prediction_manifest, selected_details, POSTPROCESS, OUTPUT_DIR / 'figures/errors')
review = val_records[0]
probabilities = np.load(prediction_manifest['entries'][review['file_name']]['path'], mmap_mode='r')
fig = plot_prediction(review, probabilities, POSTPROCESS, OUTPUT_DIR / 'figures/prediction_review.png')
plt.show(); plt.close(fig)
print('Saved review figures:', gallery)
""")
    md("""## 27. Test inference

Use the selected checkpoint and validation-selected parameters. Each original
test image receives one full-resolution forward pass, distributed across both
T4s. Four CPU jobs overlap reading, GPU forwards, postprocessing and RLE encoding.
Per-image atomic journals
support resuming an interruption and record zero detections explicitly. Journals
are bound to checkpoint, postprocessing, normalization and source file content.
""")
    code("""if RUN_TEST_INFERENCE and effective_cfg.smoke:
    raise ValueError('Smoke checkpoints cannot create a final competition submission.')
test_paths = discover_test_images(DATA_ROOT, workers=cfg.workers) if RUN_TEST_INFERENCE else []
def predict_test_instances(path):
    probabilities = predictor(path)
    return extract_instances(probabilities.astype(np.float16), POSTPROCESS)[1]
print('Test sources to process:', len(test_paths))
""")
    md("""## 28. Submission generation

One row per non-empty predicted filament: `filament_id,segmentation_rle`.
IDs retain the original test stem and add `_1`, `_2`, … . Values are compressed
COCO RLE counts only, encoded from Fortran-order 2048-square masks. Standard CSV
escaping is used; no quotation characters are inserted into values. Zero detected
filaments means no row, plus an explicit completed entry in the sidecar manifest.

```text
native independent masks → Fortran-order uint8 → pycocotools encode → counts string
 → original_stem_suffix → per-image journal → atomic submission.csv
```
""")
    code("""if RUN_TEST_INFERENCE:
    submission_report = generate_submission(test_paths, predict_test_instances, OUTPUT_DIR / 'submission.csv',
        OUTPUT_DIR / 'inference_journal', PROVENANCE | {'postprocess':POSTPROCESS}, workers=cfg.workers)
    print(submission_report)
else:
    print('Final submission is disabled in smoke mode.')
""")
    md("""## 29. Submission validation

Verify every RLE by decode/re-encode, exact schema, unique IDs, original stem
mapping, binary non-empty masks, duplicate masks, and complete source accounting
against the inference manifest. Shape is fixed to 2048 × 2048. Never add fabricated
empty-instance rows for missing detections. The standalone synthetic check below
also exercises the RLE path in smoke mode.
""")
    code("""rle_test = np.zeros((32,32), bool); rle_test[4:22,8:13] = True
assert np.array_equal(decode_rle_to_mask(encode_mask_to_rle(rle_test), (32,32)), rle_test)
if RUN_TEST_INFERENCE:
    import json
    sidecar = json.loads((OUTPUT_DIR / 'submission.manifest.json').read_text())
    print(validate_submission(OUTPUT_DIR / 'submission.csv', [p.stem for p in test_paths],
        image_counts=sidecar['image_counts']))
else:
    print('Synthetic COCO RLE round trip passed; no final CSV created in smoke mode.')
""")
    md("""## 30. Results summary, reproducibility and limitations

Inspect the selected checkpoint, `effective_config.json`, split, runtime report,
epoch logs, target fingerprints, validation reports, search CSV/JSON and inference
manifest. Archive them together. Full training-dependent artifacts exist only
after the relevant successful stage; smoke outputs are separate.

Limitations: two pixel heads do not encode overlapping instance membership;
watershed can fragment long structures; conservative reconnection can overmerge;
the default single-annotator validation policy underrepresents disagreement;
postprocessing search uses float16 maps; no temporal/date split is assumed;
source grouping prevents exact duplicate leakage but not all correlated dates.
Two-T4 NCCL performance and competition quality must be measured in Kaggle.

Recommended controlled experiments: width 8/16/32 after real memory probing,
separation radius, sparse-positive weighting, watershed seed spacing, grouped
cross-validation including multiple annotators, and only then optional clDice.
Never use test ground truth or public-leaderboard selection for these choices.
""")
    code("""summary = {'mode':MODE, 'checkpoint':str(CHECKPOINT), 'checkpoint_sha256':file_hash(CHECKPOINT),
    'validation':selected_summary, 'postprocess':POSTPROCESS, 'test_submission_created':RUN_TEST_INFERENCE,
    'warning':'Smoke validation is an execution check, not a performance estimate.' if cfg.smoke else
              'Validation reused for model/postprocess selection; confirm on an untouched grouped holdout.'}
atomic_json(OUTPUT_DIR / 'results_summary.json', summary)
print(summary)
""")
    nb.cells=cells
    nb.metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
        language_info=dict(name="python",version="3.12"),
        kaggle=dict(accelerator="gpu",isInternetEnabled=True),
        filament=dict(source_files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    for i,cell in enumerate(nb.cells): cell.id=f"filament-{i:03d}"
    nbf.validate(nb)
    output=ROOT/'solar_filament_unet_pq.ipynb'
    with output.open('w',encoding='utf8',newline='\n') as handle:
        nbf.write(nb,handle)
    print(f"Wrote {output}: {len(cells)} cells, {output.stat().st_size:,} bytes")


if __name__=='__main__': build()
