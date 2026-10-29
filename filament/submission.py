"""Counts-only COCO RLE CSVs with resumable per-source inference journals."""
import csv
import json
import os
import uuid
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from .io import atomic_json, fingerprint
from .parallel import ordered_map

COLUMNS = ["filament_id", "segmentation_rle"]


def encode_mask_to_rle(mask: np.ndarray) -> str:
    array=np.asarray(mask)
    if array.ndim!=2 or not np.isin(array,[0,1]).all(): raise ValueError("Expected binary HW mask")
    if not array.any(): raise ValueError("Empty predicted instances must be omitted")
    return mask_utils.encode(np.asfortranarray(array,dtype=np.uint8))["counts"].decode("ascii")


def decode_rle_to_mask(counts: str, shape=(2048,2048)) -> np.ndarray:
    if not isinstance(counts,str) or not counts or any(ord(c)<48 or ord(c)>111 for c in counts):
        raise ValueError("Invalid compressed COCO RLE counts")
    decoded=mask_utils.decode(dict(size=list(shape),counts=counts.encode("ascii")))
    if decoded.shape!=tuple(shape) or not np.isin(decoded,[0,1]).all(): raise ValueError("Invalid decoded RLE")
    # Canonical round trip also catches truncated/overlong run descriptions.
    canonical=mask_utils.encode(np.asfortranarray(decoded))["counts"].decode("ascii")
    if canonical!=counts: raise ValueError("Non-canonical or corrupt RLE")
    return decoded.astype(bool)


def build_filament_id(image_stem: str, index: int) -> str:
    if not image_stem or index<1: raise ValueError("Need original image stem and positive suffix")
    return f"{image_stem}_{index}"


def build_submission_rows(image_stem: str, instances: list, shape=(2048,2048)) -> list[dict]:
    rows=[]
    for i,mask in enumerate(instances,1):
        if mask.shape!=tuple(shape): raise ValueError("Submission mask dimensions differ from original image")
        counts=encode_mask_to_rle(mask)
        if not np.array_equal(decode_rle_to_mask(counts,shape),mask): raise ValueError("RLE round trip failed")
        rows.append(dict(filament_id=build_filament_id(image_stem,i),segmentation_rle=counts))
    return rows


def generate_submission(test_paths: list, predict_instances, output_path, journal_dir,
                        provenance: dict, shape=(2048,2048), workers=0) -> dict:
    """Bounded parallel inference/RLE journals; one deterministic atomic CSV writer.

    predict_instances(path) returns independent bool masks. Journals are bound to
    checkpoint/postprocessing/preprocessing AND test file content signatures by
    the caller. Zero detections have an explicit completed journal with no rows.
    """
    from .io import file_hash
    paths=sorted(map(Path,test_paths))
    if len({p.stem for p in paths})!=len(paths): raise ValueError("Duplicate test stems")
    identity=fingerprint(dict(provenance=provenance,shape=shape))
    journal_dir=Path(journal_dir)/identity
    journal_dir.mkdir(parents=True,exist_ok=True)
    image_counts={}
    output=Path(output_path); output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+f".{uuid.uuid4().hex}.tmp")
    def process_source(p):
        source_hash=file_hash(p)
        journal=journal_dir/(fingerprint(p.stem)+".json")
        result=json.loads(journal.read_text(encoding="utf8")) if journal.exists() else None
        if result is None or result.get("source_hash")!=source_hash:
            rows=build_submission_rows(p.stem,predict_instances(p),shape)
            result=dict(image_stem=p.stem,source_hash=source_hash,rows=rows)
            atomic_json(journal,result)
        if result["image_stem"]!=p.stem: raise ValueError("Journal image mismatch")
        for row in result["rows"]:
            if not decode_rle_to_mask(row["segmentation_rle"],shape).any(): raise ValueError("Empty journal mask")
        return p.stem,result["rows"]
    try:
        with temporary.open("w",newline="",encoding="utf8") as handle:
            writer=csv.DictWriter(handle,fieldnames=COLUMNS); writer.writeheader()
            for stem,rows in ordered_map(process_source,paths,workers):
                writer.writerows(rows)
                image_counts[stem]=len(rows)
        # Validate before making the final artifact visible.
        report=validate_submission(temporary,[p.stem for p in paths],shape,image_counts)
        os.replace(temporary,output)
        atomic_json(output.with_suffix(".manifest.json"),dict(provenance=provenance,
            journal_identity=identity,image_counts=image_counts,shape=list(shape),validation=report))
        return report
    finally: temporary.unlink(missing_ok=True)


def validate_submission(path, image_stems: list, shape=(2048,2048), image_counts=None) -> dict:
    """Stream all rows; no fake rows are permitted for empty images."""
    allowed=set(image_stems)
    if len(allowed)!=len(image_stems): raise ValueError("Duplicate source IDs")
    seen, masks_seen, actual=set(),set(),{s:0 for s in allowed}
    with open(path,newline="",encoding="utf8") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=COLUMNS: raise ValueError(f"Submission columns must be exactly {COLUMNS}")
        for row in reader:
            if set(row)!=set(COLUMNS) or None in row.values(): raise ValueError("Malformed CSV row")
            fid=row["filament_id"]
            try: stem,suffix=fid.rsplit("_",1)
            except ValueError: raise ValueError("Missing filament suffix") from None
            if stem not in allowed or not suffix.isdigit() or int(suffix)<1: raise ValueError(f"Invalid image ID: {fid}")
            if fid in seen: raise ValueError("Duplicate filament ID")
            seen.add(fid)
            mask=decode_rle_to_mask(row["segmentation_rle"],shape)
            if not mask.any(): raise ValueError("Empty submission mask")
            key=(stem,row["segmentation_rle"])
            if key in masks_seen: raise ValueError("Duplicate instance mask")
            masks_seen.add(key); actual[stem]+=1
    if image_counts is not None and actual!=image_counts: raise ValueError("Missing or unexpected inference records")
    return dict(rows=len(seen),images=len(allowed),zero_detection_images=sum(v==0 for v in actual.values()),
                all_rles_verified=True,coverage_verified=image_counts is not None)


# Original notebook names remain available for callers.
mask_to_rle=encode_mask_to_rle
create_submission=generate_submission
