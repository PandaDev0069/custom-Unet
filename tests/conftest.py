import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

torch.set_num_threads(1)
cv2.setNumThreads(1)


def annotation(mask,index="a"):
    rle=mask_utils.encode(np.asfortranarray(mask,dtype=np.uint8)); rle["counts"]=rle["counts"].decode()
    return dict(id=index,segmentation=rle)


@pytest.fixture
def record_factory(tmp_path):
    def make(masks,name="source",size=32):
        from filament.targets import build_foreground_mask
        path=tmp_path/f"{name}.jpeg"
        image=(build_foreground_mask(masks,(size,size))*190+20).astype(np.uint8)
        cv2.imwrite(str(path),image)
        return dict(image_id=name,file_name=path.name,image_path=str(path),height=size,width=size,
            source_hash=name,annotation_ids=[str(i) for i in range(len(masks))],
            annotations=[annotation(m,str(i)) for i,m in enumerate(masks)])
    return make
