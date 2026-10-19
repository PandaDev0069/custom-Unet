"""Organizer-compatible PQ: all IoU>0.5 pairs, global annotator-record aggregation.

No Hungarian assignment: overlapping GT can legitimately match the same prediction
under the supplied notebook. RLE overlap avoids dense N x H x W float stacks.
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def _rles(layers):
    if hasattr(layers, "detach"): layers = layers.detach().cpu().numpy()
    return [mask_utils.encode(np.asfortranarray(np.asarray(m, dtype=np.uint8))) for m in layers]


def get_overlap_matrices(gt_layers, pred_layers) -> tuple[np.ndarray, np.ndarray]:
    """Independent masks -> float64 GT-by-prediction IoU and Dice arrays."""
    gt, pred = _rles(gt_layers), _rles(pred_layers)
    if not gt or not pred:
        return np.zeros((len(gt),len(pred))), np.zeros((len(gt),len(pred)))
    iou = np.asarray(mask_utils.iou(gt, pred, [0]*len(pred)), dtype=np.float64)
    return iou, 2*iou/(1+iou)


def compute_tp_fp_fn(iou_matrix: np.ndarray, threshold=.5) -> dict:
    """Strict > threshold; TP counts pairs, FP/FN count unmatched columns/rows."""
    matrix = np.asarray(iou_matrix)
    if matrix.ndim != 2: raise ValueError("Expected GT x prediction overlap matrix")
    hit = matrix > threshold
    return dict(tp=int(hit.sum()), fp=int((~hit.any(axis=0)).sum()),
        fn=int((~hit.any(axis=1)).sum()), iou_sum=float(matrix[hit].sum()),
        matched_iou=matrix[hit].tolist())


def summarize_validation_results(results: list[dict]) -> dict:
    totals = {k:sum(r[k] for r in results) for k in ("tp","fp","fn","iou_sum")}
    denominator = totals["tp"] + .5*totals["fp"] + .5*totals["fn"]
    return dict(**totals, pq=totals["iou_sum"]/denominator if denominator else 0.,
        matched_instance_iou=totals["iou_sum"]/totals["tp"] if totals["tp"] else 0., records=len(results),
        predicted_instances=sum(r.get("n_pred",0) for r in results),
        fragmented_gt=sum(r.get("fragmented_gt",0) for r in results),
        merged_predictions=sum(r.get("merged_predictions",0) for r in results))


def evaluate_iou_distribution(iou_matrix) -> list:
    matrix = np.asarray(iou_matrix)
    return matrix[matrix > 0].tolist()


def evaluate_dice_distribution(iou_matrix) -> list:
    return [2*x/(1+x) for x in evaluate_iou_distribution(iou_matrix)]


def analyze_fragmentation(iou_matrix) -> dict:
    degrees = (np.asarray(iou_matrix)>0).sum(axis=1)
    return dict(fragmented_gt=int((degrees>1).sum()), gt_overlap_degrees=degrees.tolist())


def analyze_overmerging(iou_matrix) -> dict:
    degrees = (np.asarray(iou_matrix)>0).sum(axis=0)
    return dict(merged_predictions=int((degrees>1).sum()), pred_overlap_degrees=degrees.tolist())


def evaluate_panoptic_quality(gt_instances, pred_instances, threshold=.5) -> dict:
    iou, dice = get_overlap_matrices(gt_instances, pred_instances)
    counts = compute_tp_fp_fn(iou, threshold)
    denominator = counts["tp"]+.5*(counts["fp"]+counts["fn"])
    return dict(**counts, pq=counts["iou_sum"]/denominator if denominator else 0.,
        n_gt=iou.shape[0], n_pred=iou.shape[1],
        iou_distribution=evaluate_iou_distribution(iou), dice_distribution=evaluate_dice_distribution(iou),
        iou_matrix=iou.tolist(), **analyze_fragmentation(iou), **analyze_overmerging(iou))


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame, shape=(2048,2048)) -> pd.DataFrame:
    """Compatibility with the official CSV helper; score once per annotator-image.

    CSV cannot describe empty annotation records. Production record evaluation
    includes them explicitly (all-empty aggregate scores zero, as official PQ).
    """
    from .submission import decode_rle_to_mask
    gt_keys = gt_df.filament_id.str.rsplit("_",n=1).str[0]
    pred_keys = pred_df.filament_id.str.rsplit("_",n=1).str[0]
    rows=[]
    for key in gt_keys.unique():
        source = key.split("-",1)[1]
        gt = [decode_rle_to_mask(x,shape) for x in gt_df.loc[gt_keys==key,"segmentation_rle"]]
        pred = [decode_rle_to_mask(x,shape) for x in pred_df.loc[pred_keys==source,"segmentation_rle"]]
        iou,dice = get_overlap_matrices(gt,pred)
        rows.append(dict(annotator_image=key,iou_matrix=iou,dice_matrix=dice,n_gt=len(gt),n_pred=len(pred)))
    return pd.DataFrame(rows,columns=["annotator_image","iou_matrix","dice_matrix","n_gt","n_pred"])


def get_pq_score(overlap_df: pd.DataFrame) -> float:
    return summarize_validation_results([compute_tp_fp_fn(r.iou_matrix) for r in overlap_df.itertuples()])["pq"]
