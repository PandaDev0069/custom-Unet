# Codex Task: Build a Complete Full-Resolution Solar Filament Instance Segmentation Pipeline

#### Note: the output notebook will be runned in kaggle, with 2 T4 GPU, 15GB VRAM, 1 core cpu, 30 Gib Ram, with around 30GB disk space. You can run the notebook for testing in this local machine with i9 cpu and 306ti GPU, 8gb VRAM, 32GB RAM, 500GB disk space.

## 1. Objective

Implement a complete, production-quality, Kaggle-compatible training and inference pipeline for the **Solar Filament Segmentation Challenge 2026**.

Competition: https://www.kaggle.com/competitions/filament-segmentation-2026/

The implementation must use a **custom PyTorch U-Net**, train on native **2048×2048 grayscale H-alpha images**, and produce **individual filament instance masks** for the official Panoptic Quality (PQ) evaluation.

The complete pipeline must include:

1. Dataset discovery, inspection, and validation.
2. COCO annotation parsing and instance-level target generation.
3. Leakage-safe train/validation splitting.
4. Efficient full-resolution preprocessing, caching, and data loading.
5. Configurable augmentation.
6. Custom U-Net architecture with multiple output channels.
7. PQ-oriented differentiable loss functions.
8. GPU-efficient training with single-GPU and multi-GPU support.
9. Checkpointing, early stopping, experiment logging, and recovery.
10. Full-resolution validation and instance-aware metrics.
11. Multiple configurable postprocessing algorithms.
12. Validation-based hyperparameter optimization.
13. Full-resolution test inference.
14. COCO RLE submission generation and verification.
15. Training graphs and qualitative visualizations.
16. Thorough notebook documentation and reproducibility instructions.

**Implement the code. Do not respond with only an architecture proposal or pseudocode.**

## 2. Existing files — inspect before implementation

The workspace contains or should contain:

- `solar-filament-u-net-full-image(1).ipynb`
- `self-evaluation-notebook.ipynb`
- `filament_pq_proxy_loss.py`
- A Python file containing my custom `ModifiedUNet`, `DoubleConv`, and associated functions.

Locate and inspect these files first.

If filenames differ, search the workspace. If a required source is unavailable, clearly identify it rather than silently inventing its contents.

### Source priority

1. Use my saved **ModifiedUNet** as the primary model architecture.
2. Reuse the custom loss functions from `filament_pq_proxy_loss.py`, refining them when technically necessary.
3. Reuse reliable infrastructure from my previous full-image notebook.
4. Follow the official self-evaluation notebook for scoring behavior.
5. Consult the Kaggle competition documentation for submission requirements and constraints.

### Reuse existing infrastructure

The previous notebook includes useful implementations of:

- `find_dataset_root()`
- `polygons_to_mask()`
- `make_records()`
- `FullImageDataset`
- `probe_batch_sizes()`
- `run_epoch()`
- `predict_full_image()`
- `component_map()`
- `mask_to_rle()`
- `create_submission()`

It also includes:

- Multi-GPU DistributedDataParallel (DDP)
- Automatic mixed precision (AMP)
- Gradient accumulation
- Gradient checkpointing
- Automatic VRAM probing
- GroupNorm and global-context architecture components
- Image-level splitting
- Disk caching
- Training logs and checkpoints
- Threshold exploration
- Submission verification

Reuse or adapt these where appropriate. Do not discard working infrastructure unnecessarily.

However, do not blindly copy incorrect behavior.

## 3. Critical issue in the previous notebook

The previous pipeline merged every annotated filament into one binary mask using `polygons_to_mask()`.

The model therefore learned foreground segmentation rather than individual instance separation.

Its PQ validation also extracted connected components from combined ground-truth masks, which is not equivalent to evaluating the original individual annotations.

**Correct both problems.**

The new implementation must preserve individual filament identities from the original COCO annotations.

Do not assume that one connected component equals one filament.

Two separate filament instances may touch or overlap, and one filament may contain disconnected regions.

Preserve the original per-instance annotation masks for evaluation. A single integer instance-ID map cannot represent overlapping instances faithfully, so handle overlaps explicitly and document the supervision policy.

Never silently delete, merge, or overwrite ground-truth instances.

## 4. Dataset discovery and integrity

Use the existing MAGFiLO dataset structure from the notebook.

Implement functions for:

- Finding the dataset root.
- Discovering train/test images.
- Loading COCO JSON annotations.
- Indexing images and annotations.
- Checking file existence and dimensions.
- Detecting duplicate source images and multiple annotators.
- Validating polygon and RLE annotations.
- Summarizing dataset statistics.

Use 2048×2048 images without resizing.

Use `pycocotools` for standards-compatible annotation decoding.

Ensure annotation coordinate ordering, masks, and original image dimensions are correct.

Generate statistics including:

- Image counts
- Instances per image
- Foreground pixel ratios
- Instance area distributions
- Small-instance frequency
- Annotation overlaps
- Duplicate/multi-annotator records

Include visual data inspection before training.

## 5. Target generation

Generate and retain:

**Input image**

`[1, 2048, 2048]`

**Foreground target**

Binary union of all annotated filament pixels.

**Individual ground-truth instances**

A collection of independent binary masks, preserving their original identities.

**Instance-ID map**

An integer label map for supervision where unambiguous.

**Boundary target**

A boundary or separation supervision map derived from individual annotations.

**Optional centerline target**

An optional skeleton or centerline representation for preserving thin structures.

Required functions should include:

- `decode_annotation_mask()`
- `load_image_instances()`
- `build_foreground_mask()`
- `build_instance_id_map()`
- `build_boundary_target()`
- `build_centerline_target()`
- `validate_targets()`
- `cache_training_targets()`

Important:

A boundary between foreground and background is different from a boundary separating two filament instances.

The implementation must distinguish these concepts.

Avoid constructing a boundary map that causes thin filaments to be completely erased during inference.

Prefer a separation-aware target and a carefully documented overlap-handling strategy.

## 6. Leakage-safe train/validation split

Split by unique source image filename, not by annotations or individual filament masks.

All annotations of the same source image must belong to the same partition.

Respect the previous notebook's handling of multi-annotator images unless a demonstrably better strategy is justified.

Make the split deterministic and save a reusable split manifest.

Suggested functions:

- `group_annotations_by_source()`
- `create_train_val_split()`
- `verify_no_leakage()`
- `save_split_manifest()`
- `load_split_manifest()`

Never use test images or test ground-truth metadata to tune model parameters.

## 7. Full-resolution preprocessing

Use native 2048×2048 resolution throughout training, validation, and inference.

No resizing to 512×512 or 1024×1024.

No tiled training or tiled inference in the primary pipeline.

Preprocessing should include:

- Reading 8-bit grayscale JPEG images.
- Converting to float32.
- Consistent intensity normalization.
- Optional robust contrast normalization.
- Stride-compatible padding only when necessary.
- Image-to-tensor conversion.

Implement configurable training augmentations:

- Horizontal and vertical flips
- 90-degree rotations
- Mild brightness/contrast changes
- Mild gamma variation
- Optional low-intensity noise

Apply geometric transformations consistently to all annotation-derived targets.

Avoid unnecessarily destructive augmentations that damage fine filament morphology.

Make preprocessing deterministic during validation and test inference.

## 8. Custom U-Net architecture

Use my saved `ModifiedUNet` implementation as the primary architecture.

It contains:

- `DoubleConv`
- Encoder with MaxPool2d stages
- Decoder with ConvTranspose2d
- Skip connections
- `_match_and_concat()` for dynamic spatial alignment
- Final convolution producing raw logits

Preserve its fundamental architecture and make it configurable.

### Required architecture

Input:

`[B, 1, 2048, 2048]`

Primary outputs:

- Channel 0: Filament foreground logits
- Channel 1: Filament boundary/separation logits

Output:

`[B, 2, 2048, 2048]`

Allow an optional third centerline head as an experimental feature, but keep the first implementation limited to two channels.

Configurable parameters:

- Input channels
- Output channels
- Initial channel width
- Normalization method
- Activation type
- Gradient checkpointing
- Optional architecture enhancements

Support GroupNorm as an alternative to BatchNorm for small batches.

Retain the ability to use the previous notebook's deeper `FullImageUNet` as an optional experimental model, but **do not silently substitute it for my saved `ModifiedUNet`**.

Maintain a single consistent model interface.

Test both even and odd image dimensions.

Document tensor dimensions throughout the encoder and decoder.

## 9. Loss function

Start from `filament_pq_proxy_loss.py`.

Reuse and examine:

- `FilamentPQProxyLoss`
- `instance_boundaries()`
- `soft_erode()`
- `soft_open()`
- `soft_skeleton()`
- `soft_cldice_loss()`

Do not blindly trust the existing implementation.

Review correctness, memory consumption, numerical stability, and compatibility with the new targets.

Implement a configurable composite objective:

**Foreground loss**
- Binary cross entropy with logits
- Soft Dice loss

**Instance separation loss**
- Boundary/separation loss
- Class weighting for sparse positive pixels

**Optional structure preservation**
- Soft clDice loss

All weights must be configurable.

Be especially careful that foreground-background boundaries are not confused with boundaries between different filament instances.

Handle ambiguous overlapping supervision explicitly, using suitable masks or exclusions when appropriate.

Use FP32 for sensitive loss calculations under AMP.

Report each individual loss term during training.

Do not claim that this loss directly optimizes official PQ. It is a differentiable surrogate.

Soft clDice can be expensive on 2048×2048 tensors. Default it to disabled until memory profiling confirms that it is practical.

## 10. Efficient GPU training

Target environment:

**Kaggle Notebook — potentially two NVIDIA Tesla T4 GPUs, approximately 15 GB VRAM each.**

The pipeline must also work with one CUDA GPU.

Use:

- PyTorch DistributedDataParallel for multiple GPUs
- NCCL backend on supported Linux CUDA environments
- Automatic mixed precision (FP16 on T4)
- Gradient scaling where appropriate
- Gradient accumulation
- Activation checkpointing
- Channels-last memory format when beneficial
- AdamW optimizer
- Configurable learning-rate scheduler
- Gradient clipping
- Early stopping
- Best-checkpoint saving
- Last-checkpoint saving
- Resume training support

### Memory optimization

Reuse and improve the existing memory-probing system.

Probe actual forward and backward passes with the new architecture and loss.

Measure:

- Allocated GPU memory
- Reserved GPU memory
- Peak memory
- Feasible microbatch size

Use realistic training paths during memory probing.

Avoid relying on inference-only measurements for training memory estimates.

Provide automatic fallback for out-of-memory microbatches.

Make initial channel count configurable, allowing smaller values if 32 channels cannot fit.

Do not reduce image resolution as an automatic fallback.

### DataLoader

Support:

- Multiprocess workers
- Pinned memory
- Persistent workers when appropriate
- Configurable prefetching
- Lazy disk loading
- Target caching
- DDP DistributedSampler
- Correct epoch-dependent shuffling

Avoid loading all 2048×2048 images and instance masks into RAM simultaneously.

Avoid creating excessive worker copies or duplicate caches.

### Training correctness

Ensure:

- Correct gradient accumulation scaling
- Correct handling of incomplete accumulation windows
- Correct DDP gradient synchronization
- Proper validation behavior
- No optimizer stepping during validation
- Appropriate `train()` and `eval()` modes
- Stable AMP handling
- Deterministic seeds where practical

## 11. Training and checkpoint management

Provide:

- `train_one_epoch()`
- `validate_one_epoch()`
- `save_checkpoint()`
- `load_checkpoint()`
- `resume_training()`
- `update_early_stopping()`
- `log_training_metrics()`

Checkpoints should preserve enough state to resume correctly:

- Model state
- Optimizer state
- Scheduler state
- AMP scaler state
- Epoch
- Best validation metrics
- Configuration
- Random generator states when practical

Log epoch duration, total training time, and GPU utilization/memory statistics.

Use validation loss for training diagnostics, but select the best final model using a clearly defined instance-aware validation objective, preferably PQ.

Since full PQ evaluation may be computationally expensive, allow configurable evaluation frequency.

## 12. Instance-aware postprocessing

This is one of the most important parts of the implementation.

The U-Net predicts foreground and separation probability maps, but the competition requires individual filament instances.

Implement a modular postprocessing system.

Required stages:

1. Foreground probability thresholding.
2. Boundary/separation probability processing.
3. Candidate instance extraction.
4. Boundary-aware instance splitting.
5. Optional watershed-based separation.
6. Small-artifact filtering.
7. Optional morphological refinement.
8. Instance-ID assignment.
9. Individual binary mask extraction.

Required algorithms:

**Method A: Connected-component baseline**

Preserve the simplest baseline for comparison.

**Method B: Boundary-aware connected components**

Use separation predictions to distinguish touching filament candidates.

**Method C: Marker-controlled watershed**

Use reliable seed selection and foreground constraints.

**Method D: Optional continuity-aware refinement**

Explore carefully constrained reconnection of fragmented structures without merging genuinely distinct filaments.

Implement each method independently behind a common interface.

Postprocessing must not erase narrow structures simply because they were classified as object boundaries.

All thresholds and filtering parameters must be configurable.

Do not assume more aggressive merging always improves PQ.

## 13. Official Panoptic Quality evaluation

Use `self-evaluation-notebook.ipynb` as the primary reference.

Inspect and reuse its relevant functions, including:

- `get_overlap_matrices()`
- `get_overlap_df()`
- `get_pq_score()`
- Distribution plotting functions
- Many-to-one and one-to-many analysis functions

The official notebook uses instance-level overlap matrices and an IoU matching threshold of 0.5.

Reproduce its semantics accurately, including grouping by image/annotator and its score aggregation.

Use the original independent annotation masks for validation.

Do not compute ground-truth instances using connected components of a union mask.

Implement:

- `evaluate_panoptic_quality()`
- `evaluate_dice_distribution()`
- `evaluate_iou_distribution()`
- `analyze_fragmentation()`
- `analyze_overmerging()`
- `compute_tp_fp_fn()`
- `summarize_validation_results()`

Track:

- PQ
- True positives
- False positives
- False negatives
- Matched-instance IoU
- Dice and IoU distributions
- One-to-many relations
- Many-to-one relations
- Predicted instance counts
- Runtime per image

Cross-check your implementation against the official evaluator using synthetic masks.

Document precisely any limitations or differences.

## 14. Postprocessing hyperparameter search

Do not retrain the model for every postprocessing experiment.

Perform one forward pass for each validation image and cache the prediction probability maps efficiently.

Reuse those predictions to test different postprocessing configurations.

Search configurable combinations of:

- Foreground threshold
- Boundary threshold
- Minimum component area
- Separation method
- Watershed parameters
- Morphological parameters

Choose settings based primarily on validation PQ.

Also report Dice/IoU and fragmentation/merging behavior.

Avoid selecting configurations based on the Kaggle public leaderboard.

Save results to a CSV and generate comparison plots.

Store the winning postprocessing configuration in a reproducible JSON file.

## 15. Visualization and error analysis

Produce publication-quality but readable visualizations.

Required visualizations:

**Dataset inspection**
- Original H-alpha image
- Ground-truth instances in different colors
- Foreground target
- Boundary target

**Training**
- Training and validation loss curves
- Individual loss components
- Learning-rate history
- GPU memory statistics

**Inference**
- Original image
- Foreground probability map
- Boundary probability map
- Binary prediction
- Individual predicted instances
- Ground-truth instance overlay

**Error analysis**
- False positive examples
- Missed filaments
- Fragmentation examples
- Over-merging examples
- Best and worst PQ images

For morphology inspection, show both complete 2048×2048 images and native-resolution zoomed regions.

Crops are allowed for visualization, not for primary model inference.

## 16. Test inference

Use the selected model checkpoint and validation-chosen inference configuration.

Perform a full-image forward pass for every test image.

Process each image independently.

Return a collection of 2048×2048 binary instance masks.

Keep the inference pipeline deterministic.

Test data must not be used for model training or ground-truth-based tuning.

Support resuming inference after interruption without corrupting output.

## 17. Kaggle submission generation

Follow the official competition specification exactly.

Produce:

`submission.csv`

Columns:

- `filament_id`
- `segmentation_rle`

Each row represents one predicted filament.

Use the original test image ID/stem followed by a unique suffix such as `_1`, `_2`, `_3`.

Use `pycocotools` compressed RLE encoding.

Only store the RLE counts string, not the size dictionary.

Ensure masks are 2048×2048.

Do not insert literal quotation characters into the RLE value; use standard CSV escaping when writing the file.

Implement:

- `encode_mask_to_rle()`
- `decode_rle_to_mask()`
- `build_filament_id()`
- `build_submission_rows()`
- `generate_submission()`
- `validate_submission()`

Validate:

- Exact column names
- Unique filament IDs
- Non-empty valid RLE values
- Correct decoded dimensions
- Correct image ID mapping
- No duplicate rows
- No invalid or empty masks
- Correct handling of images with zero detected filaments

Perform RLE round-trip tests.

## 18. Testing

Create unit tests and integration tests.

At minimum test:

- Annotation decoding
- Instance-target generation
- Overlapping annotations
- Empty masks
- Touching separate filaments
- Single filament with disconnected regions
- Train/validation leakage prevention
- Odd-dimensional U-Net inputs
- Two-channel model output shapes
- Loss forward and backward
- AMP compatibility
- Postprocessing
- PQ calculation
- RLE encoding/decoding
- Submission schema

Use small synthetic images for automated tests.

Include a GPU smoke test that verifies the actual training path when a GPU is available.

A full Kaggle training run is not required during development, but do not claim it was executed if it was not.

## 19. Notebook requirements

Create:

`solar_filament_unet_pq.ipynb`

It must be a complete, well-organized, Kaggle-ready notebook.

Required section order:

1. Project introduction and competition objective
2. Competition evaluation and PQ mathematics
3. Pipeline architecture and flow diagrams
4. Dependencies and runtime checks
5. Imports and configuration
6. Dataset discovery
7. Annotation inspection
8. Dataset statistics
9. Train/validation splitting
10. Target generation and validation
11. Full-resolution preprocessing
12. Augmentation
13. Dataset and DataLoader implementation
14. Custom U-Net architecture
15. Loss functions and mathematical explanations
16. GPU setup and memory probing
17. Training configuration
18. Training execution
19. Training visualization
20. Checkpoint loading
21. Full-resolution validation inference
22. Instance extraction and postprocessing
23. Official PQ evaluation
24. Dice/IoU/fragmentation analysis
25. Postprocessing parameter optimization
26. Qualitative prediction review
27. Test inference
28. Submission generation
29. Submission validation
30. Results summary and limitations

### Notebook portability

The notebook must run on Kaggle without requiring manually uploaded custom Python modules beyond the official competition dataset.

Keep reusable source modules in the repository, but ensure the standalone notebook contains or generates everything required to execute.

If multi-GPU DDP requires a separate worker script, generate it automatically from notebook cells and launch it safely.

Keep notebook cells reasonably sized and readable.

Avoid embedding huge, unreadable code strings where a cleaner generated-module approach is possible.

Make all major operations configurable.

Provide a small smoke-test mode and a full-training mode.

## 20. Documentation requirements

After implementing the notebook, produce complete documentation.

Create:

- `README.md`
- `docs/PIPELINE.md`
- `docs/MODEL_ARCHITECTURE.md`
- `docs/LOSS_FUNCTIONS.md`
- `docs/TRAINING.md`
- `docs/EVALUATION.md`
- `docs/POSTPROCESSING.md`
- `docs/FUNCTION_REFERENCE.md`
- `docs/EXPERIMENTS.md`
- `requirements.txt`

### Documentation quality

Explain:

- What each pipeline stage does
- Why it is necessary
- How each stage works
- All important function inputs and outputs
- Important tensor shapes
- How the mathematics connects to the implementation
- Memory usage and performance considerations
- How to modify the architecture and training settings
- How to introduce new losses
- How to add postprocessing algorithms
- How to evaluate and compare experiments
- How to reproduce the submission

Include diagrams for:

1. Complete training/inference workflow
2. Dataset and annotation transformation
3. U-Net encoder/decoder and skip connections
4. Multi-head outputs
5. Composite loss
6. DDP training
7. Instance extraction
8. Official PQ evaluation
9. Submission generation

Use Mermaid diagrams in Markdown documents and readable equivalent diagrams in the notebook where necessary.

Write documentation for someone learning computer vision and PyTorch who wants to understand and modify the implementation independently.

Do not just describe what the code does. Explain why important design choices were made.

In particular, document the differences between the previous binary-segmentation implementation and the new instance-aware approach.

## 21. Code quality

Use:

- Modular Python functions and classes
- Type hints
- Clear docstrings
- Meaningful names
- Central configuration
- Explicit shape validation
- Useful exceptions
- Reproducible experiment settings
- Efficient tensor operations
- Comments explaining non-obvious logic

Avoid unnecessary abstractions or excessive comments explaining trivial lines.

Preserve proven components from the original notebook while removing duplication and incorrect assumptions.

Ensure all imports, functions, parameters, and data formats are consistent across modules.

## 22. Required deliverables

Produce:

**Primary artifact**
- `solar_filament_unet_pq.ipynb`

**Implementation**
- Reusable model, loss, dataset, training, evaluation, postprocessing, and submission modules

**Documentation**
- All Markdown documentation files listed above

**Validation**
- Unit tests
- Integration tests
- Smoke-test instructions
- Requirements with package versions

**Results/artifact organization**
- Checkpoints
- Logs
- Training graphs
- Validation reports
- Postprocessing comparison results
- `submission.csv`

Generated training-dependent artifacts should only be produced when the corresponding execution succeeds.

## 23. Implementation workflow

Complete the work in this order:

**Phase 1:** Inspect all existing code, identify reusable components and known problems.

**Phase 2:** Implement and test dataset indexing, instance masks, boundary targets, and split integrity.

**Phase 3:** Integrate the saved custom U-Net and loss functions.

**Phase 4:** Implement and test the GPU-efficient training engine.

**Phase 5:** Implement the instance-aware evaluation and compare it with the official self-evaluation notebook.

**Phase 6:** Implement postprocessing algorithms and validation-based parameter search.

**Phase 7:** Implement test inference and RLE submission generation.

**Phase 8:** Assemble the complete executable Kaggle notebook.

**Phase 9:** Run available tests, verify notebook structure and execution order, and fix errors.

**Phase 10:** Generate complete technical documentation.

Do not stop after planning.

Do not replace requested implementation with placeholder functions.

Do not skip tests merely because the complete dataset or Kaggle GPU is unavailable.

Use synthetic tests and static validation where full execution is impossible.

## 24. Final acceptance criteria

The implementation is complete only when:

- The custom U-Net is integrated and supports native-resolution inference.
- Original ground-truth filament instances remain available individually.
- Foreground and separation targets are constructed correctly.
- The loss function can run forward and backward.
- The training engine supports Kaggle GPU environments.
- DDP, AMP, checkpointing, and memory probing are correctly integrated.
- Validation evaluates individual instances using the official PQ methodology.
- Postprocessing produces independent filament masks.
- Multiple postprocessing strategies can be compared without retraining.
- Submission generation follows the official format.
- Synthetic tests and applicable smoke tests pass.
- The notebook is organized and independently runnable.
- All required technical documentation exists.

At completion, report:

1. Files created or modified.
2. Original functions reused.
3. Functions replaced and reasons.
4. Tests actually performed and their results.
5. Parts that could not be tested without Kaggle hardware or data.
6. Known limitations and recommended experiments.

**Primary priority: correctness of individual-instance segmentation and official PQ evaluation.**

**Secondary priority: memory-efficient full-resolution training and inference.**

**Third priority: modularity, reproducibility, and documentation.**

Do not claim improved competition performance until measured against the validation dataset.

Begin by inspecting the existing files, then implement the complete system.
