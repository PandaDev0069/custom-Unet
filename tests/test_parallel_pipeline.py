"""Concurrency checks and serial/parallel parity for the notebook runtime."""
import threading

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from filament.data import (cache_training_targets, summarize_dataset, FullImageDataset,
                           TrainValidationDataset, seed_loader_worker)
from filament.inference import (MultiDevicePredictor, cache_validation_predictions,
                                evaluate_cached_predictions, search_postprocessing)
from filament.parallel import ordered_map
from filament.submission import generate_submission
from filament.training import PhaseSampler
from filament.model import build_model


def test_bounded_ordered_parallel_work():
    barrier=threading.Barrier(4)
    consumed=[]
    def inputs():
        for index in range(12):
            consumed.append(index)
            yield index
    def job(index):
        if index<4: barrier.wait(timeout=10)
        return index*index
    results=ordered_map(job,inputs(),workers=4)
    assert next(results)==0
    assert consumed==list(range(4))  # No unbounded queued images/results.
    assert list(results)==[i*i for i in range(1,12)]
    def fail(index):
        if index==2: raise ValueError("worker failure")
        return index
    with pytest.raises(ValueError,match="worker failure"):
        list(ordered_map(fail,range(8),workers=4))


class GuardedModel(torch.nn.Module):
    def __init__(self, barrier):
        super().__init__()
        self.guard=threading.Lock()
        self.barrier=barrier
        self.calls=0

    def forward(self, inputs):
        assert self.guard.acquire(blocking=False), "Concurrent use of one GPU model"
        try:
            self.calls+=1
            if self.calls==1: self.barrier.wait(timeout=10)
            return torch.cat((inputs*8-4, inputs*0-8),dim=1)
        finally:
            self.guard.release()


def test_parallel_preparation_inference_search_submission(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[4:25,9:15]=True
    original=[record_factory([mask],f"image{i}") for i in range(4)]
    serial=cache_training_targets(original,tmp_path/"serial")
    parallel=cache_training_targets(original,tmp_path/"parallel",workers=4)
    assert summarize_dataset(original)==summarize_dataset(original,workers=4)
    for first,second in zip(serial,parallel):
        assert first["target_fingerprint"]==second["target_fingerprint"]
        with np.load(first["target_path"]) as a, np.load(second["target_path"]) as b:
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
    barrier=threading.Barrier(2)
    models=[GuardedModel(barrier),GuardedModel(barrier)]
    options=dict(expected_size=32,amp=False,channels_last=False)
    predictor=MultiDevicePredictor([(model,"cpu") for model in models],**options)
    manifest=cache_validation_predictions(models[0],parallel,tmp_path/"predictions","cpu",
        dict(checkpoint_sha256="synthetic"),workers=4,predictor=predictor,**options)
    assert all(model.calls>0 for model in models)  # Both exclusive device slots run concurrently.
    assert len(manifest["entries"])==4
    config=dict(method="components",min_area=1)
    summary,details=evaluate_cached_predictions(parallel,manifest,config)
    concurrent_summary,concurrent_details=evaluate_cached_predictions(parallel,manifest,config,workers=4)
    assert summary==concurrent_summary
    assert [r["image_id"] for r in details]==[r["image_id"] for r in concurrent_details]
    grid=dict(method=["components","boundary"],foreground_threshold=[.4,.6])
    first,serial_trials=search_postprocessing(parallel,manifest,grid,tmp_path/"search1",config)
    second,parallel_trials=search_postprocessing(parallel,manifest,grid,tmp_path/"search4",config,workers=4)
    assert first==second  # Includes first-wins tie handling.
    assert serial_trials.drop(columns="seconds").equals(parallel_trials.drop(columns="seconds"))
    paths=[r["image_path"] for r in parallel]
    reports=[]
    for workers in (0,4):
        reports.append(generate_submission(paths,lambda _: [mask],tmp_path/f"submission{workers}.csv",
            tmp_path/f"journal{workers}",dict(checkpoint="synthetic"),(32,32),workers=workers))
    assert reports[0]==reports[1]
    assert (tmp_path/"submission0.csv").read_bytes()==(tmp_path/"submission4.csv").read_bytes()
    def fail(_): raise AssertionError("Valid journal must skip inference")
    assert generate_submission(paths,fail,tmp_path/"resumed.csv",tmp_path/"journal4",
        dict(checkpoint="synthetic"),(32,32),workers=4)==reports[0]
    with pytest.raises(ValueError,match="provenance"):
        cache_validation_predictions(models[0],parallel,tmp_path/"bad","cpu",{},predictor=predictor)


def test_persistent_pool_switches_train_validation_and_epoch(record_factory,tmp_path):
    mask=np.zeros((32,32),bool); mask[3:24,7:11]=True
    records=cache_training_targets([record_factory([mask],f"phase{i}") for i in range(4)],
                                   tmp_path/"targets",workers=4)
    train=FullImageDataset(records[:3],augment=True,seed=42)
    validation=FullImageDataset(records[3:])
    dataset=TrainValidationDataset(train,validation)
    sampler=PhaseSampler(DistributedSampler(train,num_replicas=1,rank=0,shuffle=False),[0])
    loader=DataLoader(dataset,batch_size=2,sampler=sampler,num_workers=2,prefetch_factor=2,
                      persistent_workers=True,multiprocessing_context="spawn",worker_init_fn=seed_loader_worker)
    try:
        for epoch in (1,7):
            train.set_epoch(epoch)
            dataset.select(False); sampler.validation=False
            batches=list(loader)
            assert [i for batch in batches for i in batch["index"].tolist()]==[0,1,2]
            assert torch.equal(batches[0]["image"][0],train[0]["image"])
            worker_pids=[worker.pid for worker in loader._iterator._workers]
            dataset.select(True); sampler.validation=True
            batches=list(loader)
            assert len(batches)==1 and batches[0]["index"].tolist()==[0]
            assert torch.equal(batches[0]["image"][0],validation[0]["image"])
            assert [worker.pid for worker in loader._iterator._workers]==worker_pids
    finally:
        if loader._iterator is not None: loader._iterator._shutdown_workers()


@pytest.mark.gpu
def test_two_gpu_prediction_matches_serial(record_factory,tmp_path):
    if torch.cuda.device_count()<2: pytest.skip("Two CUDA GPUs required")
    mask=np.zeros((32,32),bool); mask[4:24,7:12]=True
    records=[record_factory([mask],f"gpu{i}") for i in range(4)]
    models=[build_model(dict(init_features=2,checkpoint_blocks=False)).to(f"cuda:{i}").eval()
            for i in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    options=dict(expected_size=32,amp=True,channels_last=True)
    predictor=MultiDevicePredictor(list(zip(models,["cuda:0","cuda:1"])),**options)
    parallel=cache_validation_predictions(models[0],records,tmp_path/"dual","cuda:0",{},
        workers=4,predictor=predictor,**options)
    serial=cache_validation_predictions(models[0],records,tmp_path/"single","cuda:0",{},**options)
    for name in parallel["entries"]:
        a=np.load(parallel["entries"][name]["path"])
        b=np.load(serial["entries"][name]["path"])
        assert np.allclose(a,b,atol=2e-3)
