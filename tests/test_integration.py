import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from filament.data import cache_training_targets,FullImageDataset
from filament.model import build_model
from filament.losses import FilamentPQProxyLoss
from filament.training import train_one_epoch,validate_one_epoch
from filament.inference import cache_validation_predictions,search_postprocessing,evaluate_cached_predictions
from filament.postprocess import extract_instances
from filament.submission import generate_submission


def test_end_to_end_synthetic(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[5:26,12:17]=1
    records=cache_training_targets([record_factory([mask],"a"),record_factory([mask],"b")],tmp_path/"targets")
    dataset=FullImageDataset(records); loader=DataLoader(dataset,batch_size=1)
    model=build_model(dict(name="modified",init_features=2,checkpoint_blocks=True))
    criterion=FilamentPQProxyLoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    train=train_one_epoch(model,loader,criterion,optimizer,scaler,"cpu",amp=False,accumulation_steps=2)
    validation,rows=validate_one_epoch(model,loader,criterion,"cpu",amp=False,pq_records=records,
        postprocess_config=dict(method="boundary",min_area=1))
    assert train["samples"]==2 and len(rows)==2 and np.isfinite(validation["loss"])
    manifest=cache_validation_predictions(model,records,tmp_path/"predictions","cpu",dict(checkpoint_sha256="synthetic"),expected_size=32,amp=False)
    best,trials=search_postprocessing(records,manifest,dict(method=["components","boundary"],foreground_threshold=[.4,.6]),tmp_path/"search",dict(min_area=1))
    assert len(trials)==4 and (tmp_path/"search/best_postprocessing.json").exists()
    paths=[records[0]["image_path"]]
    probabilities=np.load(manifest["entries"][records[0]["file_name"]]["path"])
    result=generate_submission(paths,lambda _:extract_instances(probabilities,best["postprocess"])[1],
        tmp_path/"submission.csv",tmp_path/"journal",best,(32,32))
    assert result["images"]==1 and result["all_rles_verified"]
    from pathlib import Path
    path=Path(records[0]["image_path"])
    path.write_bytes(path.read_bytes()+b"changed-source")
    with pytest.raises(ValueError,match="changed after validation caching"):
        evaluate_cached_predictions(records,manifest,best["postprocess"])


def test_persistent_worker_epoch_state(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[4:20,8:12]=1
    records=cache_training_targets([record_factory([mask],"worker")],tmp_path/"cache_worker")
    dataset=FullImageDataset(records,augment=True,seed=42)
    loader=DataLoader(dataset,batch_size=1,num_workers=1,persistent_workers=True,multiprocessing_context="spawn")
    first=next(iter(loader))["image"].clone()
    dataset.set_epoch(7)
    second=next(iter(loader))["image"]
    assert not torch.equal(first,second)
    assert torch.equal(second[0],dataset[0]["image"])
    del loader


def test_validation_loss_and_pq_timings(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[4:25,10:14]=True
    records=cache_training_targets([record_factory([mask],f"timing{i}") for i in range(3)],tmp_path/"timings")
    loader=DataLoader(FullImageDataset(records),batch_size=2)
    class FixedMaps(torch.nn.Module):
        def forward(self,x): return torch.cat((x*20-10,torch.full_like(x,-10)),dim=1)
    model=FixedMaps(); criterion=FilamentPQProxyLoss()
    scored,details=validate_one_epoch(model,loader,criterion,"cpu",amp=False,pq_records=records,
        postprocess_config=dict(method="components",min_area=1))
    loss_only,no_details=validate_one_epoch(model,loader,criterion,"cpu",amp=False)
    assert scored["pq"]==1 and len(details)==3
    assert scored["postprocess_seconds"]>=0 and scored["pq_scoring_seconds"]>=0
    assert scored["seconds"]>=scored["pq_scoring_seconds"]
    assert scored["loss"]==pytest.approx(loss_only["loss"])
    assert loss_only["samples"]==3 and no_details==[] and "pq" not in loss_only
