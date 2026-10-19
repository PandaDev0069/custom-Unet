import csv
import numpy as np
import pandas as pd
import pytest
import torch
import reference_official as official
from filament.evaluation import *
from filament.postprocess import extract_instances
from filament.submission import *


@pytest.mark.parametrize("kind",["perfect","touching","overlap","disconnected","half","empty_gt","empty_pred","both_empty"])
def test_official_evaluator_parity(kind):
    a=np.zeros((32,32),bool); a[4:20,4:10]=1
    b=np.zeros_like(a); b[4:20,10:16]=1
    gt=[a,b]; pred=[a,b]
    if kind=="overlap": gt=[a,a.copy()]; pred=[a]
    if kind=="disconnected": gt=[a|b]; pred=[a,b]
    if kind=="half": gt=[a|b]; pred=[a]
    if kind=="empty_gt": gt=[]
    if kind=="empty_pred": pred=[]
    if kind=="both_empty": gt=[]; pred=[]
    gt_tensor=torch.tensor(np.stack(gt) if gt else np.zeros((0,32,32)),dtype=torch.float32)
    pred_tensor=torch.tensor(np.stack(pred) if pred else np.zeros((0,32,32)),dtype=torch.float32)
    ref_iou,ref_dice=official.get_overlap_matrices(gt_tensor,pred_tensor)
    iou,dice=get_overlap_matrices(gt,pred)
    assert np.allclose(iou,ref_iou.numpy()) and np.allclose(dice,ref_dice.numpy())
    reference=official.get_pq_score(pd.DataFrame([dict(iou_matrix=ref_iou,n_gt=len(gt),n_pred=len(pred))]))
    assert evaluate_panoptic_quality(gt,pred)["pq"]==pytest.approx(reference)
    if kind=="overlap": assert evaluate_panoptic_quality(gt,pred)["tp"]==2
    if kind=="half": assert evaluate_panoptic_quality(gt,pred)["tp"]==0


def test_grouping_and_global_aggregation():
    a=np.zeros((16,16),bool); a[2:10,3:7]=1
    b=np.zeros_like(a); b[10:14,12:15]=1
    gt=pd.DataFrame([dict(filament_id=f"{key}_1",segmentation_rle=encode_mask_to_rle(mask))
        for key,mask in [("ann1-image",a),("ann2-image",b)]])
    pred=pd.DataFrame([dict(filament_id="image_1",segmentation_rle=encode_mask_to_rle(a))])
    result=get_overlap_df(gt,pred,(16,16))
    original=official.rles_to_layers
    official.rles_to_layers=lambda rles: original(rles,16,16)
    try: reference=official.get_overlap_df(gt,pred)
    finally: official.rles_to_layers=original
    assert len(result)==2 and result.n_pred.tolist()==[1,1]
    assert get_pq_score(result)==pytest.approx(official.get_pq_score(reference))
    assert get_pq_score(result)==pytest.approx(.5)


@pytest.mark.parametrize("method",["components","boundary","watershed","continuity"])
def test_postprocess_preserves_thin_foreground(method):
    p=np.zeros((2,40,40),np.float32)
    p[0,2:38,20]=.9; p[1,2:38,20]=.95
    labels,masks=extract_instances(p,dict(method=method,min_area=1,seed_min_area=1))
    assert len(masks)==1 and np.array_equal(labels>0,p[0]>.5)
    p=np.zeros((2,40,40),np.float32); p[0,5:35,10:30]=.9; p[1,5:35,19:21]=.99
    labels,masks=extract_instances(p,dict(method=method,min_area=1,seed_min_area=1,seed_distance=8))
    assert np.array_equal(labels>0,p[0]>.5)
    if method in ("boundary","continuity"): assert len(masks)==2
    assert extract_instances(np.zeros_like(p),dict(method=method))[1]==[]


def test_rle_schema_and_resume(tmp_path):
    a=np.zeros((16,16),bool); a[2:12,4:8]=1
    counts=encode_mask_to_rle(a)
    assert np.array_equal(decode_rle_to_mask(counts,(16,16)),a)
    with pytest.raises(ValueError): encode_mask_to_rle(np.zeros_like(a))
    paths=[tmp_path/"image_with_underscore.jpeg",tmp_path/"zero.jpeg"]
    for p in paths: p.write_bytes(b"source")
    calls=[]
    def predict(p): calls.append(p.stem); return [] if p.stem=="zero" else [a]
    output=tmp_path/"submission.csv"
    report=generate_submission(paths,predict,output,tmp_path/"journal",dict(model="x"),shape=(16,16))
    assert report["rows"]==1 and report["zero_detection_images"]==1 and report["coverage_verified"]
    generate_submission(paths,predict,output,tmp_path/"journal",dict(model="x"),shape=(16,16))
    assert len(calls)==2
    generate_submission(paths,predict,output,tmp_path/"journal",dict(model="changed"),shape=(16,16))
    assert len(calls)==4
    frame=pd.read_csv(output); pd.concat([frame,frame]).to_csv(output,index=False)
    with pytest.raises(ValueError): validate_submission(output,[p.stem for p in paths],(16,16))


def test_interrupted_inference_recovers(tmp_path):
    a=np.zeros((8,8),bool); a[1:4,1:4]=1
    paths=[tmp_path/"a.jpg",tmp_path/"b.jpg"]
    for p in paths: p.write_bytes(b"x")
    def fail(p):
        if p.stem=="b": raise RuntimeError("interrupted")
        return [a]
    output=tmp_path/"submission.csv"
    with pytest.raises(RuntimeError): generate_submission(paths,fail,output,tmp_path/"journal",{},(8,8))
    assert not output.exists()
    calls=[]
    def resume(p): calls.append(p.stem); return [a]
    generate_submission(paths,resume,output,tmp_path/"journal",{},(8,8))
    assert calls==["b"]
