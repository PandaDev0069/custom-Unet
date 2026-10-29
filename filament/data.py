"""Dataset integrity, grouped splitting, compact targets and lazy full-image loading."""
import hashlib
import json
import os
import uuid
from threading import Lock
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io import atomic_json, file_hash, fingerprint
from .targets import build_targets, load_image_instances
from .parallel import ordered_map

ANNOTATION_FILENAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TARGET_VERSION = "independent-coco-separation-8connected-v2"


def find_dataset_root(override: str = "") -> Path:
    """Find existing MAGFiLO layout locally or in nested Kaggle inputs."""
    candidates = [Path(override).expanduser()] if override else []
    if not override:
        for parent in (Path.cwd(), *Path.cwd().parents):
            candidates.extend([parent, parent / "MAGFilo_1.0_Kaggle_2026", parent / "MAGFiLO_1.0_Kaggle_2026"])
        if Path("/kaggle/input").exists():
            candidates.extend(p.parent.parent for p in Path("/kaggle/input").rglob(ANNOTATION_FILENAME))
    for root in candidates:
        if ((root / "train" / ANNOTATION_FILENAME).is_file() and
            (root / "train/train_images").is_dir() and (root / "test/test_images").is_dir()):
            return root.resolve()
    raise FileNotFoundError("Set data_root to the MAGFiLO directory containing train/ and test/")


def discover_test_images(root: Path, expected_size: int = 2048, workers=0) -> list[Path]:
    paths = sorted(p.resolve() for p in (Path(root) / "test/test_images").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths: raise ValueError("No test images discovered")
    if len({p.stem for p in paths}) != len(paths): raise ValueError("Duplicate test stems")
    def validate(path):
        with Image.open(path) as im:
            if im.size != (expected_size, expected_size): raise ValueError(f"Non-native test image {path}")
    for _ in ordered_map(validate, paths, workers): pass
    return paths


def make_records(root: Path, expected_size: int = 2048, hash_images: bool = True, workers=0) -> tuple[list, dict]:
    """Index annotation records; validate images once and fingerprint decoded pixels.

    Multiple annotators remain separate records. Pixel-identical filenames are
    grouped together for splitting even if JPEG metadata differs. No test labels
    are loaded. Annotation payloads stay in the parent process only after caching.
    """
    root = Path(root)
    annotation_path = root / "train" / ANNOTATION_FILENAME
    data = json.loads(annotation_path.read_text(encoding="utf8"))
    ids = [str(im["id"]) for im in data["images"]]
    if len(set(ids)) != len(ids): raise ValueError("Duplicate COCO image IDs")
    annotation_ids = [str(a["id"]) for a in data["annotations"]]
    if len(set(annotation_ids)) != len(annotation_ids): raise ValueError("Duplicate annotation IDs")
    by_image = defaultdict(list)
    known = set(ids)
    for a in data["annotations"]:
        if str(a["image_id"]) not in known: raise ValueError("Annotation references unknown image")
        if a.get("iscrowd", 0): raise ValueError("Crowd annotation requires an explicit evaluation policy")
        by_image[str(a["image_id"])].append(a)
    files, records = {}, []
    image_dir = (root / "train/train_images").resolve()
    def inspect(im):
        name = im["file_name"]
        path = (image_dir / name).resolve()
        if not path.is_relative_to(image_dir): raise ValueError("Image path escapes dataset")
        shape = (int(im["height"]), int(im["width"]))
        if shape != (expected_size, expected_size): raise ValueError(f"Non-native annotation dimensions: {name}")
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as opened:
            if opened.size != shape[::-1]: raise ValueError(f"JPEG and annotation dimensions differ: {name}")
        pixels = read_grayscale(path, shape)
        return name, hashlib.sha256(pixels.tobytes()).hexdigest() if hash_images else name
    unique = {im["file_name"]: im for im in data["images"]}
    files.update(ordered_map(inspect, unique.values(), workers))
    for im in data["images"]:
        name = im["file_name"]
        path = (image_dir / name).resolve()
        if not path.is_relative_to(image_dir): raise ValueError("Image path escapes dataset")
        shape = (int(im["height"]), int(im["width"]))
        if shape != (expected_size, expected_size): raise ValueError(f"Non-native annotation dimensions: {name}")
        records.append(dict(image_id=str(im["id"]), file_name=name, image_path=str(path),
            source_hash=files[name], height=shape[0], width=shape[1], annotations=by_image[str(im["id"])],
            annotation_ids=[str(a["id"]) for a in by_image[str(im["id"])]]))
    groups = group_annotations_by_source(records)
    hashes = defaultdict(list)
    for name, digest in files.items(): hashes[digest].append(name)
    report = dict(annotation_sha256=file_hash(annotation_path), annotation_records=len(records),
        source_images=len(files), annotations=len(annotation_ids),
        multi_annotator_sources={k:len(v) for k,v in groups.items() if len(v)>1},
        duplicate_pixel_sources=[v for v in hashes.values() if len(v)>1],
        instances_per_record=dict(Counter(len(r["annotations"]) for r in records)),
        hash_images=hash_images)
    return records, report


def read_grayscale(path, expected_shape=None) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None: raise OSError(f"Cannot decode image {path}")
    if im.dtype != np.uint8 or im.ndim != 2: raise ValueError("Expected 8-bit grayscale")
    if expected_shape is not None and tuple(im.shape) != tuple(expected_shape): raise ValueError("Image dimensions changed")
    return im


def group_annotations_by_source(records: list) -> dict:
    groups = defaultdict(list)
    for record in records: groups[record["file_name"]].append(record)
    return dict(groups)


def create_train_val_split(records: list, val_fraction=.2, seed=10,
                           multi_annotator_train_only=True) -> dict:
    """Keep prior multi-annotator-in-training policy; group duplicate pixel sources."""
    groups = defaultdict(list)
    for r in records: groups[r.get("source_hash", r["file_name"])].append(r)
    file_groups = group_annotations_by_source(records)
    eligible = sorted(key for key, rows in groups.items() if not multi_annotator_train_only or
        all(len(file_groups[r["file_name"]]) == 1 for r in rows))
    if len(eligible) < 2: raise ValueError("Need at least two eligible independent source groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    desired = max(1, round(val_fraction * len(file_groups)))
    selected, count = set(), 0
    for key in eligible[:-1]:
        if count >= desired: break
        selected.add(key)
        count += len({r["file_name"] for r in groups[key]})
    train, val = [], []
    for key, rows in groups.items(): (val if key in selected else train).extend(rows)
    verify_no_leakage(train, val)
    return dict(train=[r["image_id"] for r in train], val=[r["image_id"] for r in val], seed=seed,
        val_fraction=val_fraction, multi_annotator_train_only=multi_annotator_train_only,
        dataset_fingerprint=record_fingerprint(records))


def record_fingerprint(records: list) -> str:
    return fingerprint([{k:r[k] for k in ("image_id", "file_name", "source_hash", "annotation_ids")}
                        for r in sorted(records, key=lambda r:r["image_id"])])


def verify_no_leakage(train: list, val: list) -> None:
    if not train or not val: raise ValueError("Empty partition")
    for key in ("image_id", "file_name", "source_hash"):
        left = {r.get(key, r["file_name"]) for r in train}
        right = {r.get(key, r["file_name"]) for r in val}
        if left & right: raise ValueError(f"Train/validation leakage by {key}")


def save_split_manifest(path, manifest: dict): atomic_json(path, manifest)


def load_split_manifest(path, records: list) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf8"))
    if result["dataset_fingerprint"] != record_fingerprint(records): raise ValueError("Split belongs to a different dataset")
    ids = {r["image_id"] for r in records}
    all_ids = result["train"] + result["val"]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != ids: raise ValueError("Invalid split membership")
    by_id = {r["image_id"]:r for r in records}
    verify_no_leakage([by_id[i] for i in result["train"]], [by_id[i] for i in result["val"]])
    return result


def cache_training_targets(records: list, cache_dir, radius=1, centerline=False, workers=0) -> list:
    """Atomic compressed per-record arrays; original COCO masks remain in compact JSON."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Identical annotations may share one target file. Serialize only colliding
    # fingerprints, including concurrent Windows replaces of the same path.
    locks = [Lock() for _ in range(64)]
    def cache_record(r):
        key = fingerprint(dict(version=TARGET_VERSION, shape=[r["height"],r["width"]],
            annotations=r["annotations"], radius=radius, centerline=centerline))
        path = cache_dir / f"{key}.npz"
        original_path = cache_dir / f"{key}.{fingerprint(r['image_id'])}.json"
        with locks[int(key[:8], 16) % len(locks)]:
            if not path.exists():
                targets = build_targets(r, radius, centerline)
                temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("wb") as handle: np.savez_compressed(handle, **targets)
                    os.replace(temporary, path)
                finally: temporary.unlink(missing_ok=True)
            atomic_json(original_path, dict(image_id=r["image_id"], height=r["height"], width=r["width"],
                annotations=r["annotations"], annotation_ids=r["annotation_ids"]))
        return {k:v for k,v in r.items() if k != "annotations"} | dict(
            target_path=str(path.resolve()), instances_path=str(original_path.resolve()), target_fingerprint=key)
    cached = []
    for index, record in enumerate(ordered_map(cache_record, records, workers)):
        cached.append(record)
        if (index+1) % 100 == 0: print(f"Cached {index+1}/{len(records)} annotation records", flush=True)
    return cached


def cached_instances(record: dict) -> list:
    payload = json.loads(Path(record["instances_path"]).read_text(encoding="utf8"))
    return load_image_instances(payload)


def summarize_dataset(records: list, small_area=20, workers=0) -> tuple[dict, list]:
    """Stream bounded parallel annotation records; retain only compact statistics."""
    def summarize_record(r):
        masks = load_image_instances(r)
        coverage = np.zeros((r["height"], r["width"]), dtype=np.uint32)
        image_areas = []
        for m in masks:
            coverage += m
            image_areas.append(int(m.sum()))
        return dict(image_id=r["image_id"], file_name=r["file_name"], instances=len(masks),
            foreground_ratio=float((coverage>0).mean()), overlap_pixels=int((coverage>1).sum()),
            small_instances=sum(a < small_area for a in image_areas)), image_areas
    rows, areas = [], []
    for index, (row, image_areas) in enumerate(ordered_map(summarize_record, records, workers)):
        rows.append(row); areas.extend(image_areas)
        if (index+1)%100==0: print(f"Validated {index+1}/{len(records)} annotation records",flush=True)
    summary = dict(records=len(rows), instances=len(areas), small_area=small_area,
        small_instances=sum(a<small_area for a in areas), area_quantiles=(
            dict(zip(["min","q25","median","q75","max"],np.quantile(areas,[0,.25,.5,.75,1]).tolist())) if areas else {}),
        overlap_records=sum(r["overlap_pixels"]>0 for r in rows), instance_areas=areas)
    return summary, rows


def normalize_image(image: np.ndarray, mode="unit") -> np.ndarray:
    values = image.astype(np.float32) / 255.
    if mode == "robust":
        lo, hi = np.percentile(values, [1, 99])
        values = np.clip((values-lo)/max(float(hi-lo), 1/255), 0, 1).astype(np.float32)
    elif mode != "unit": raise ValueError("Unknown normalization")
    return values


def augment_sample(image: np.ndarray, targets: dict, rng, options: dict) -> tuple:
    """Same geometry for all maps; mild photometric changes affect input only."""
    arrays = {"image":image, **targets}
    if options.get("flips", True):
        for axis in (0, 1):
            if rng.random() < .5: arrays = {k:np.flip(v, axis) for k,v in arrays.items()}
    if options.get("rotate90", True):
        turns = int(rng.integers(4))
        arrays = {k:np.rot90(v, turns) for k,v in arrays.items()}
    image = arrays.pop("image")
    contrast, brightness = options.get("contrast", .1), options.get("brightness", .05)
    image = np.clip((image-.5)*rng.uniform(1-contrast,1+contrast)+.5+rng.uniform(-brightness,brightness),0,1)
    gamma = options.get("gamma", .1)
    image = image ** rng.uniform(1-gamma,1+gamma)
    if options.get("noise", 0): image += rng.normal(0,options["noise"],image.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(image,0,1),dtype=np.float32), arrays


class FullImageDataset(Dataset):
    """Lazy native images and cached targets. Augmentation is indexed by epoch/item.

    Shared epoch state also reaches persistent workers. Dense original instances
    are deliberately loaded separately during evaluation, never collated.
    """
    def __init__(self, records: list, augment=False, seed=10, normalization="unit", augment_options=None):
        self.records, self.augment, self.seed = records, augment, seed
        self.normalization, self.options = normalization, augment_options or {}
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch): self._epoch.fill_(epoch)
    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = normalize_image(read_grayscale(r["image_path"], (r["height"],r["width"])), self.normalization)
        with np.load(r["target_path"], allow_pickle=False) as cached:
            targets = {k:cached[k] for k in ("foreground", "separation", "separation_valid")}
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,int(self._epoch),index]))
            image, targets = augment_sample(image, targets, rng, self.options)
        return {"image":torch.from_numpy(np.ascontiguousarray(image[None])),
            **{k:torch.from_numpy(np.ascontiguousarray(v[None])).float() for k,v in targets.items()}, "index":index}


class TrainValidationDataset(Dataset):
    """Share a single persistent worker pool across sequential train/val phases."""
    def __init__(self, train, validation):
        self.train, self.validation = train, validation
        self._validation = torch.zeros((), dtype=torch.int64).share_memory_()

    def select(self, validation): self._validation.fill_(int(validation))
    def __len__(self): return len(self.validation if self._validation.item() else self.train)
    def __getitem__(self, index):
        return (self.validation if self._validation.item() else self.train)[index]


def seed_loader_worker(worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
