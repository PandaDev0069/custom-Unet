"""COCO instance identities survive overlaps and disconnected components."""
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi


def decode_annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    """Decode polygons (x,y), compressed RLE or uncompressed COCO RLE as bool HW."""
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        if not segmentation: raise ValueError(f"Empty polygon annotation {annotation.get('id')}")
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=float)
            if points.ndim != 1 or len(points) < 6 or len(points) % 2 or not np.isfinite(points).all():
                raise ValueError(f"Invalid polygon: {annotation.get('id')}")
            xy = points.reshape(-1, 2)
            if (xy < 0).any() or (xy[:, 0] > width).any() or (xy[:, 1] > height).any():
                raise ValueError(f"Polygon outside image: {annotation.get('id')}")
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        if list(segmentation.get("size", [])) != [height, width]: raise ValueError("RLE size mismatch")
        rle = dict(segmentation)
        counts = rle.get("counts")
        if isinstance(counts, list):
            if any(not isinstance(c, int) or c < 0 for c in counts) or sum(counts) != height*width:
                raise ValueError("Invalid uncompressed RLE run lengths")
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(counts, str): rle["counts"] = counts.encode("ascii")
        elif not isinstance(counts, bytes): raise ValueError("Invalid RLE counts")
    else: raise ValueError(f"Missing/invalid segmentation: {annotation.get('id')}")
    result = mask_utils.decode(rle)
    if result.shape != (height, width): raise ValueError("Decoded annotation has incorrect shape")
    result = result.astype(bool)
    if not result.any(): raise ValueError(f"Empty annotation: {annotation.get('id')}; fix the source explicitly")
    return result


def load_image_instances(record: dict) -> list[np.ndarray]:
    """One mask per original annotation; never split by connected components."""
    return [decode_annotation_mask(a, record["height"], record["width"]) for a in record["annotations"]]


def build_foreground_mask(instances: list, shape: tuple) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for mask in instances: result |= mask
    return result


def build_instance_id_map(instances: list, shape: tuple) -> np.ndarray:
    """0 background, positive annotation ordinal, -1 ambiguous overlap."""
    ids = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(instances, 1):
        overlap = mask & (ids != 0)
        ids[mask & (ids == 0)] = index
        ids[overlap] = -1
    return ids


def build_boundary_target(instances: list, shape: tuple, radius: int = 1) -> tuple:
    """Inter-instance separation, not exterior contours; return separation and valid.

    Foreground within radius of TWO instances is positive. Overlap and its
    radius-neighborhood are excluded from separation BCE. Union remains supervised.
    """
    if radius < 1: raise ValueError("Separation radius must be positive")
    coverage, nearby = np.zeros(shape, np.uint32), np.zeros(shape, np.uint32)
    for mask in instances:
        coverage += mask
        nearby += ndi.binary_dilation(mask, structure=np.ones((3,3)), iterations=radius)
    ambiguous = coverage > 1
    valid = ~ndi.binary_dilation(ambiguous, structure=np.ones((3,3)), iterations=radius) if ambiguous.any() else ~ambiguous
    return (coverage > 0) & (nearby > 1) & valid, valid


def build_centerline_target(instances: list, shape: tuple) -> np.ndarray:
    from skimage.morphology import skeletonize
    result = np.zeros(shape, dtype=bool)
    for mask in instances: result |= skeletonize(mask)
    return result


def validate_targets(targets: dict, instances: list) -> None:
    shape = targets["foreground"].shape
    for key in ("foreground", "separation", "separation_valid", "instance_ids"):
        if targets[key].shape != shape: raise ValueError(f"Misaligned {key}")
    fg = targets["foreground"].astype(bool)
    if not np.array_equal(fg, build_foreground_mask(instances, shape)): raise ValueError("Incorrect foreground union")
    ids = targets["instance_ids"]
    if not np.array_equal(ids != 0, fg): raise ValueError("IDs lost foreground")
    if (targets["separation"] & ~fg).any(): raise ValueError("Separation outside foreground")
    if (targets["separation"] & ~targets["separation_valid"].astype(bool)).any(): raise ValueError("Ambiguous separation labels")
    if not np.array_equal(ids, build_instance_id_map(instances, shape)): raise ValueError("Instance identities changed")


def build_targets(record: dict, radius=1, centerline=False) -> dict:
    instances = load_image_instances(record)
    shape = (record["height"], record["width"])
    separation, valid = build_boundary_target(instances, shape, radius)
    result = dict(foreground=build_foreground_mask(instances, shape), instance_ids=build_instance_id_map(instances, shape),
                  separation=separation, separation_valid=valid)
    if centerline: result["centerline"] = build_centerline_target(instances, shape)
    validate_targets(result, instances)
    return result
