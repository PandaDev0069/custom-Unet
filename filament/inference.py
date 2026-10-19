"""Native inference, provenance-bound probability caches and validation searches."""
import itertools
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_grayscale, normalize_image, cached_instances, group_annotations_by_source
from .evaluation import evaluate_panoptic_quality, summarize_validation_results
from .io import atomic_json, file_hash, fingerprint
from .postprocess import extract_instances, PostprocessConfig


def predict_full_image(model, image: np.ndarray, device, amp=True, channels_last=True,
                       normalization="unit", expected_size=2048) -> np.ndarray:
    """One complete native uint8 HW image -> two complete float32 probability maps."""
    if image.dtype!=np.uint8 or image.shape!=(expected_size,expected_size):
        raise ValueError("Prediction requires a complete native uint8 grayscale image")
    device=torch.device(device)
    inputs=torch.from_numpy(normalize_image(image,normalization)[None,None]).to(device)
    if channels_last: inputs=inputs.contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=amp and device.type=="cuda"):
        logits=model(inputs)
    if logits.shape!=(1,2,expected_size,expected_size): raise ValueError("Unexpected model output shape")
    return logits[0].float().sigmoid().cpu().numpy()


def cache_validation_predictions(model, records, cache_dir, device, provenance, **predict_options) -> dict:
    """One forward pass per source; float16 NPY maps load using mmap during search."""
    namespace=fingerprint(dict(provenance=provenance,predict_options=predict_options))
    directory=Path(cache_dir)/namespace; directory.mkdir(parents=True,exist_ok=True)
    manifest=dict(provenance=provenance,predict_options=predict_options,entries={})
    for name,rows in group_annotations_by_source(records).items():
        r=rows[0]
        source_hash=file_hash(r["image_path"])
        path=directory/(fingerprint(dict(name=name,source_hash=source_hash))+".npy")
        started=time.perf_counter()
        cache_hit=path.exists()
        if not cache_hit:
            probabilities=predict_full_image(model,read_grayscale(r["image_path"]),device,**predict_options)
            temporary=path.with_name(path.name+f".{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as handle: np.save(handle,probabilities.astype(np.float16),allow_pickle=False)
                os.replace(temporary,path)
            finally: temporary.unlink(missing_ok=True)
        with path.open("rb") as handle:
            array=np.load(handle,allow_pickle=False)
            if array.shape!=(2,r["height"],r["width"]) or not np.isfinite(array).all(): raise ValueError("Corrupt prediction cache")
        manifest["entries"][name]=dict(path=str(path.resolve()),source_hash=source_hash,
            prediction_seconds=None if cache_hit else time.perf_counter()-started,cache_hit=cache_hit)
    atomic_json(directory/"manifest.json",manifest)
    return manifest


def evaluate_cached_predictions(records, manifest, postprocess_config) -> tuple[dict,list]:
    rows=[]
    for name,annotation_records in group_annotations_by_source(records).items():
        entry=manifest["entries"][name]
        if file_hash(annotation_records[0]["image_path"])!=entry["source_hash"]:
            raise ValueError(f"Source image changed after validation caching: {name}")
        probabilities=np.load(entry["path"],mmap_mode="r",allow_pickle=False)
        started=time.perf_counter()
        _,instances=extract_instances(probabilities,postprocess_config)
        seconds=time.perf_counter()-started
        for r in annotation_records:
            result=evaluate_panoptic_quality(cached_instances(r),instances)
            rows.append(dict(**result,image_id=r["image_id"],file_name=name,
                postprocess_seconds=seconds,prediction_seconds=entry["prediction_seconds"]))
    return summarize_validation_results(rows),rows


def search_postprocessing(records, manifest, grid: dict, output_dir, base_config=None) -> tuple[dict,pd.DataFrame]:
    """Tune validation-only postprocessing; predictions are never recomputed."""
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    base=dict(base_config or {})
    combinations=itertools.product(*(grid[k] for k in grid))
    trials=[]; best=None
    for index,values in enumerate(combinations):
        cfg=PostprocessConfig(**(base | dict(zip(grid,values))))
        started=time.perf_counter()
        summary,details=evaluate_cached_predictions(records,manifest,asdict(cfg))
        ious=[v for r in details for v in r["iou_distribution"]]
        dices=[v for r in details for v in r["dice_distribution"]]
        row=dict(trial=index,**asdict(cfg),**summary,mean_positive_iou=float(np.mean(ious)) if ious else 0.,
                 mean_positive_dice=float(np.mean(dices)) if dices else 0.,seconds=time.perf_counter()-started)
        trials.append(row)
        pd.DataFrame(trials).to_csv(output/"postprocessing_trials.csv",index=False)
        # Deterministic tie break: first configuration wins.
        if best is None or summary["pq"]>best["validation"]["pq"]:
            best=dict(postprocess=asdict(cfg),validation=summary,trial=index,
                prediction_provenance=manifest["provenance"],validation_image_ids=[r["image_id"] for r in records])
            atomic_json(output/"best_postprocessing.json",best)
            atomic_json(output/"best_validation_details.json",details)
        print(f"Postprocess trial {index}: PQ={summary['pq']:.5f}",flush=True)
    if best is None: raise ValueError("Empty postprocessing grid")
    frame=pd.DataFrame(trials)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(frame.trial,frame.pq,"o-")
    ax.set(xlabel="Validation trial",ylabel="Official-semantics PQ",title="Postprocessing comparison")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"postprocessing_comparison.png",dpi=180); plt.close(fig)
    return best,frame
