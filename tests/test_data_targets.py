import json
import numpy as np
import pytest
from pycocotools import mask as mask_utils
from filament.targets import *
from filament.data import *
from conftest import annotation


def test_decode_polygon_and_rle():
    polygon=[[2.,3.,9.,3.,9.,11.,2.,11.]]
    expected=mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygon,16,17))).astype(bool)
    assert np.array_equal(decode_annotation_mask(dict(segmentation=polygon),16,17),expected)
    assert np.array_equal(decode_annotation_mask(annotation(expected),16,17),expected)
    flat=expected.ravel(order="F"); runs=[]; last=0; count=0
    for value in flat:
        if value!=last: runs.append(count); count=0; last=int(value)
        count+=1
    runs.append(count)
    assert np.array_equal(decode_annotation_mask(dict(segmentation=dict(size=[16,17],counts=runs)),16,17),expected)
    for bad in ([],[[0,0,1]],[[0,0,20,0,1,1]]):
        with pytest.raises(ValueError): decode_annotation_mask(dict(segmentation=bad),16,17)
    with pytest.raises(ValueError): decode_annotation_mask(annotation(np.zeros((16,17))),16,17)


def test_overlap_touching_disconnected_empty(record_factory):
    a=np.zeros((32,32),bool); a[5:20,5:8]=1; a[25:27,3:5]=1
    b=np.zeros_like(a); b[5:20,8:11]=1
    r=record_factory([a,b]); t=build_targets(r)
    assert len(load_image_instances(r))==2
    assert (t["instance_ids"][25:27,3:5]==1).all()
    assert t["separation"][5:20,7:9].all()
    assert not t["separation"][5:20,5].any()
    assert not build_boundary_target([a],a.shape)[0].any()
    b[:,6:9]=a[:,5:8]
    overlap=build_targets(record_factory([a,b],"overlap"))
    assert (overlap["instance_ids"][a&b]==-1).all()
    assert not overlap["separation_valid"][a&b].any()
    assert np.array_equal(overlap["foreground"],a|b)
    empty=build_targets(record_factory([],"empty"))
    assert empty["foreground"].sum()==0 and empty["separation_valid"].all()
    diagonal=np.zeros_like(a); diagonal[4,4]=1
    diagonal2=np.zeros_like(a); diagonal2[5,5]=1
    sep,_=build_boundary_target([diagonal,diagonal2],a.shape)
    assert sep[4,4] and sep[5,5]


def test_split_persistence_and_duplicates(tmp_path):
    records=[dict(image_id=str(i),file_name=f"{i}.jpeg",source_hash=str(i),annotation_ids=[]) for i in range(10)]
    records.append(records[0]|dict(image_id="annotator2"))
    records.append(records[1]|dict(image_id="duplicate",file_name="copy.jpeg"))
    manifest=create_train_val_split(records,seed=2)
    assert "0" in manifest["train"] and "annotator2" in manifest["train"]
    assert ("1" in manifest["val"])==("duplicate" in manifest["val"])
    assert create_train_val_split(records,seed=2)==manifest
    path=tmp_path/"split.json"; save_split_manifest(path,manifest)
    assert load_split_manifest(path,records)==manifest
    with pytest.raises(ValueError): verify_no_leakage(records[:2],records[1:])
    with pytest.raises(ValueError): load_split_manifest(path,records[:-1])


def test_cache_augmentation_and_dataset(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[8:25,4:8]=1
    r=record_factory([mask])
    cached=cache_training_targets([r],tmp_path/"cache")
    assert len(cached_instances(cached[0]))==1
    dataset=FullImageDataset(cached,True,seed=5,augment_options=dict(brightness=0,contrast=0,gamma=0))
    first=dataset[0]
    assert first["image"].shape==(1,32,32)
    assert np.array_equal(first["image"].numpy()>.4,first["foreground"].numpy().astype(bool))
    assert all(np.array_equal(first[k],dataset[0][k]) for k in ("image","foreground","separation"))
    changed=cache_training_targets([r],tmp_path/"cache",radius=2)
    assert changed[0]["target_path"]!=cached[0]["target_path"]
