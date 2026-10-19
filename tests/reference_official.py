# Organizer functions extracted unchanged from the provided self-evaluation notebook.
import torch
import pandas as pd
import numpy as np
from pycocotools import mask as mask_util


def get_overlap_matrices(
        gt_layers: torch.Tensor,
        pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise IoU and Dice scores between ground-truth (GT) and predicted mask layers.
    This is called for all segmentations corresponding to one image.

    Args:
        gt_layers: Ground-truth layers, shape (n_gt, H, W), values in {0, 1}.
        pred_layers: Prediction layers, shape (n_pred, H, W), values in {0, 1}.

    Returns:
        A tuple ``(iou_matrix, dice_matrix)`` of tensors, each of shape
        (n_gt, n_pred) with values in [0, 1], where:

            iou_matrix[i, j]  = IoU(GT layer i, predicted layer j)
            dice_matrix[i, j] = Dice(GT layer i, predicted layer j)

        - **Rows (i, size n_gt)**: ground-truth filaments
        - **Columns (j, size n_pred)**: predicted filaments

        Example for n_gt=2 GT filaments and n_pred=3 predictions:

            iou_matrix = [[IoU(gt_0, pred_0), IoU(gt_0, pred_1), IoU(gt_0, pred_2)],
                          [IoU(gt_1, pred_0), IoU(gt_1, pred_1), IoU(gt_1, pred_2)]]
    """
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())

    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)

    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        intersection / union,
    )

    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )

    return iou_matrix, dice_matrix

def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-positive predictions for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A predicted filament is a false positive if it does not hit any GT filament:
    the sum of its column is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # column sums: [0, 0, 2]  ->  columns 0 and 1 are FPs
        fp_count_hit(hit_matrix) -> 2

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of predictions with no qualifying GT overlap.
    """
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return (gt_matches_per_pred == 0).sum().item()

def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-negative annotations for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A GT filament is a false negative if it does not hit any predicted filament: the sum of
    its row is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # row sums: [1, 0, 1]  ->  row 1 is FN
        fn_count_hit(hit_matrix) -> 1

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of annotations with no qualifying prediction overlap.
    """
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return (pred_matches_per_gt == 0).sum().item()

def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decode a list of compressed COCO RLE strings into an (n_masks, H, W) binary mask stack.

    Args:
        rles: Compressed COCO RLE strings, one per filament.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        float32 array of shape (n_masks, H, W) with values in {0, 1}, where
        n_masks is ``len(rles)``. Returns an empty (0, H, W) array if ``rles``
        is empty.
    """
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    # decode -> (H, W, n_masks), transpose to (n_masks, H, W)
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)

def process_entry(
        annotator_image: str,
        gt_annotator_image_ids: pd.Series,
        pred_image_ids: pd.Series,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Matches one annotator-image entry between prediction and GT and computes
    the overlap matrices between its GT and predicted filaments.

    Args:
        annotator_image: Annotator-image identifier in the form ``"<annotator_id>-<image_id>"``.
        gt_annotator_image_ids: Series of annotator-image identifiers, one per row of ``gt_df``.
        pred_image_ids: Series of image identifiers, one per row of ``pred_df``.
        gt_df: Ground-truth filaments; columns ``filament_id`` and ``segmentation_rle``.
        pred_df: Predicted filaments; same columns as ``gt_df``.

    Returns:
        Tuple ``(iou_matrix, dice_matrix, n_gt, n_pred)`` where the matrices
        have shape (n_gt, n_pred).
    """
    image_id = annotator_image.split("-", maxsplit=1)[1]

    # (n_gt, H, W) stack with one layer per GT filament of this annotator-image
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))

    # (n_pred, H, W) stack with one layer per predicted filament of this image
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))

    n_gt = len(gt_rles)
    n_pred = len(pred_rles)

    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)

    return iou_matrix, dice_matrix, n_gt, n_pred

def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-annotator-image table of GT vs. prediction overlap matrices.

    Args:
        gt_df: Ground-truth filaments; columns ``filament_id`` and
            ``segmentation_rle``.
        pred_df: Predicted filaments; same columns as ``gt_df``.

    Returns:
        DataFrame with one row per annotator-image and columns:

        - ``annotator_image``: annotator-image identifier
        - ``iou_matrix``: IoU tensor of shape (n_gt, n_pred)
        - ``dice_matrix``: Dice tensor of shape (n_gt, n_pred)
        - ``n_gt``: number of GT filaments in the image
        - ``n_pred``: number of predicted filaments in the image
    """
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()

    overlap_df = pd.DataFrame({"annotator_image": annotator_images})

    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )

    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df[
            "n_pred"] = zip(*results)

    return overlap_df

def get_pq_score(overlap_df: pd.DataFrame) -> float:
    """
    Compute the Panoptic Quality (PQ) score over all images.

    A GT/prediction pair is a true positive (TP) when its IoU exceeds the
    IoU threshold (0.5). Predictions with no match are false positives (FP),
    GT filaments with no match are false negatives (FN), and:

        PQ = sum(IoU of TP pairs) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

    Args:
        overlap_df: Per-image overlap table as produced by ``get_overlap_df``;
            must contain columns ``iou_matrix`` (IoU tensor of shape
            (n_gt, n_pred)), ``n_gt`` (GT filament count), and ``n_pred``
            (predicted filament count).

    Returns:
        Panoptic Quality score in ``[0, 1]`` (higher is better), or ``0.0`` if
        the denominator is zero.
    """
    iou_threshold: float = 0.5

    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue

        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())

        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count

    if denominator > 0:
        pq_score = sum(tp_iou_scores) / denominator
    else:
        pq_score = 0.0

    return pq_score