"""AMP/DDP training, real-backward memory probes and epoch-boundary recovery."""
import gc
import importlib.metadata
import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .config import Config
from .data import FullImageDataset, TrainValidationDataset, cached_instances, seed_loader_worker
from .evaluation import evaluate_panoptic_quality, summarize_validation_results
from .io import atomic_json
from .losses import FilamentPQProxyLoss
from .model import build_model
from .postprocess import extract_instances

LOSS_NAMES=("loss","bce","dice","boundary","cldice")


class PhaseSampler(Sampler):
    """Switch only after exhausting an epoch iterator; keep one loader worker pool."""
    def __init__(self, train_sampler, validation_indices):
        self.train_sampler, self.validation_indices = train_sampler, validation_indices
        self.validation = False

    def __iter__(self):
        return iter(self.validation_indices if self.validation else self.train_sampler)

    def __len__(self):
        return len(self.validation_indices if self.validation else self.train_sampler)


def seed_everything(seed: int, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=not deterministic
    # warn_only permits operations without deterministic CUDA kernels to report it.
    torch.use_deterministic_algorithms(deterministic,warn_only=True)


def rng_state() -> dict:
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state["cuda"].cpu())


def _distributed(): return dist.is_available() and dist.is_initialized()


def should_compute_pq(epoch, epochs, pq_every):
    """Always evaluate the first/final epoch, then the requested PQ cadence."""
    return epoch==1 or epoch==epochs or epoch%pq_every==0


def gather_objects(value):
    if not _distributed(): return [value]
    gathered=[None]*dist.get_world_size(); dist.all_gather_object(gathered,value)
    return gathered


def gpu_statistics(device) -> dict:
    if torch.device(device).type!="cuda": return dict(allocated_gib=0.,reserved_gib=0.,peak_allocated_gib=0.,peak_reserved_gib=0.,utilization_percent=None)
    util=None
    try: util=float(torch.cuda.utilization(device))
    except Exception:
        # Optional NVML telemetry must not abort training on an unsupported host.
        pass
    return dict(allocated_gib=torch.cuda.memory_allocated(device)/2**30,
        reserved_gib=torch.cuda.memory_reserved(device)/2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,utilization_percent=util)


def _device_batch(batch,device,channels_last):
    # For grayscale N1HW, contiguous(channels_last) is a no-op: it leaves
    # channel stride H*W. cuDNN then returns an NCHW first-convolution gradient,
    # while DDP expects the NHWC parameter stride. `to` explicitly restrides
    # singleton dimensions, and combining it with H2D avoids a second GPU copy.
    return {k:v.to(device,non_blocking=True,**(
        dict(memory_format=torch.channels_last) if k=="image" and channels_last else {}))
        for k,v in batch.items() if k!="index"}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              accumulation_steps=1, amp=True, channels_last=True, grad_clip=1.,
              pq_records=None, postprocess_config=None, log_every=20) -> tuple[dict,list]:
    """Sample-weighted accumulation including short final batches/windows.

    DDP ranks have equal training sampler lengths. Validation bypasses DDP forward
    collectives and uses disjoint non-padding samplers, including empty ranks.
    """
    training=optimizer is not None
    model.train(training)
    forward_model=model if training or not isinstance(model,DDP) else model.module
    device=torch.device(device)
    scaler=scaler or torch.amp.GradScaler("cuda",enabled=amp and device.type=="cuda")
    totals=torch.zeros(len(LOSS_NAMES)+1,dtype=torch.float64,device=device)
    rows=[]; started=time.perf_counter()
    postprocess_seconds=score_seconds=0.
    if training: optimizer.zero_grad(set_to_none=True)
    batches=len(loader)
    batch_size=loader.batch_size
    sample_count=len(loader.sampler)
    if loader.drop_last: sample_count=(sample_count//batch_size)*batch_size
    with (torch.enable_grad() if training else torch.inference_mode()):
        for index,batch in enumerate(loader):
            inputs=_device_batch(batch,device,channels_last)
            count=inputs["image"].shape[0]
            window_start=(index//accumulation_steps)*accumulation_steps
            window_end=min(window_start+accumulation_steps,batches)
            window_samples=min(sample_count,window_end*batch_size)-window_start*batch_size
            sync=index+1==window_end
            context=model.no_sync() if training and isinstance(model,DDP) and not sync else nullcontext()
            with context:
                with torch.autocast(device.type,dtype=torch.float16,enabled=amp and device.type=="cuda"):
                    logits=forward_model(inputs["image"])
                values=criterion(logits,inputs)
                if not torch.isfinite(values["loss"]): raise FloatingPointError("Nonfinite loss")
                if training: scaler.scale(values["loss"]*(count/window_samples)).backward()
            if training and sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            for i,name in enumerate(LOSS_NAMES): totals[i]+=values[name].detach().double()*count
            totals[-1]+=count
            if not training and pq_records is not None:
                probabilities=logits.float().sigmoid().cpu().numpy()
                for local,record_index in enumerate(batch["index"].tolist()):
                    r=pq_records[record_index]
                    before=time.perf_counter()
                    _,instances=extract_instances(probabilities[local],postprocess_config)
                    after_postprocess=time.perf_counter()
                    row=evaluate_panoptic_quality(cached_instances(r),instances)
                    after_score=time.perf_counter()
                    postprocess_seconds+=after_postprocess-before
                    score_seconds+=after_score-after_postprocess
                    rows.append(dict(**row,image_id=r["image_id"],file_name=r["file_name"],
                        postprocess_evaluation_seconds=time.perf_counter()-before))
            if (index+1)%log_every==0 and (not _distributed() or dist.get_rank()==0):
                print(f"{'Training' if training else 'Validation'} native batch {index+1}/{batches}",flush=True)
            del inputs,logits,values
    if device.type=="cuda": torch.cuda.synchronize(device)
    if _distributed(): dist.all_reduce(totals)
    packed=totals.cpu().tolist(); samples=packed[-1]
    metrics={name:packed[i]/max(samples,1) for i,name in enumerate(LOSS_NAMES)}
    rows=[row for rank_rows in gather_objects(rows) for row in rank_rows]
    if pq_records is not None: metrics.update(summarize_validation_results(rows))
    # Include distributed result gathering and global PQ aggregation in the
    # phase wall time, so validation timings do not hide reporting overhead.
    timings=torch.tensor([time.perf_counter()-started,postprocess_seconds,score_seconds],dtype=torch.float64,device=device)
    if _distributed(): dist.all_reduce(timings,op=dist.ReduceOp.MAX)
    seconds,postprocess_seconds,score_seconds=timings.cpu().tolist()
    metrics.update(samples=samples,seconds=seconds,images_per_second=samples/max(seconds,1e-9))
    if pq_records is not None:
        metrics.update(postprocess_seconds=postprocess_seconds,pq_scoring_seconds=score_seconds)
    return metrics,rows


def train_one_epoch(model,loader,criterion,optimizer,scaler,device,**kwargs):
    return run_epoch(model,loader,criterion,device,optimizer,scaler,**kwargs)[0]


def validate_one_epoch(model,loader,criterion,device,**kwargs):
    return run_epoch(model,loader,criterion,device,**kwargs)


def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,best_metrics,config,
                    early_stopping,history,rank_rng_states,split_fingerprint):
    """Atomic trusted checkpoint, including rank-local random streams."""
    module=model.module if isinstance(model,DDP) else model
    payload=dict(model=module.state_dict(),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict() if scheduler else None,scaler=scaler.state_dict(),epoch=epoch,
        best_metrics=best_metrics,config=config,early_stopping=early_stopping,history=history,
        rank_rng_states=rank_rng_states,world_size=len(rank_rng_states),split_fingerprint=split_fingerprint)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{uuid.uuid4().hex}.tmp")
    try: torch.save(payload,temporary); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def load_checkpoint(path,model=None,optimizer=None,scheduler=None,scaler=None,map_location="cpu") -> dict:
    # Checkpoints are locally generated, trusted artifacts with numpy RNG tuples.
    state=torch.load(path,map_location=map_location,weights_only=False)
    if model is not None:
        (model.module if isinstance(model,DDP) else model).load_state_dict(state["model"])
    for object_,key in ((optimizer,"optimizer"),(scheduler,"scheduler"),(scaler,"scaler")):
        if object_ is not None and state[key] is not None: object_.load_state_dict(state[key])
    return state


def resume_training(path,model,optimizer,scheduler,scaler,device,rank=0) -> dict:
    state=load_checkpoint(path,model,optimizer,scheduler,scaler,device)
    if rank>=len(state["rank_rng_states"]): raise ValueError("Resume requires the original DDP world size")
    restore_rng(state["rank_rng_states"][rank])
    return state


def update_early_stopping(state: dict, pq: float, patience: int, delta=1e-6) -> tuple[dict,bool,bool]:
    improved=pq>state.get("best_pq",-1)+delta
    result=dict(best_pq=pq if improved else state.get("best_pq",-1),
                stale_evaluations=0 if improved else state.get("stale_evaluations",0)+1)
    return result,improved,result["stale_evaluations"]>=patience


def log_training_metrics(path,history):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    pd.DataFrame(history).to_csv(temporary,index=False); os.replace(temporary,path)


def probe_batch_sizes(model,criterion,sample,config,device) -> tuple[int|None,list]:
    """Actual model + FP32 loss + accumulated backward + AdamW slot allocation.

    Each GPU probes unwrapped, then all ranks agree on a common fit. Model state
    and RNG are restored, so probes never become undocumented training updates.
    """
    device=torch.device(device)
    if device.type!="cuda": return 1,[dict(batch_size=1,fits=True,device="cpu",measured=False)]
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    random_state=rng_state()
    requested=config["microbatch"]
    candidates=sorted(set(config["batch_candidates"] if requested=="auto" else [1,int(requested)]))
    reports=[]
    rank=dist.get_rank() if _distributed() else 0
    try:
        for batch_size in candidates:
            before_probe=time.perf_counter()
            print(f"[rank {rank}] Probing native microbatch={batch_size}...",flush=True)
            optimizer=values=inputs=logits=scaler=None
            model.load_state_dict(original); restore_rng(random_state); model.train()
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            report=dict(batch_size=batch_size,fits=False)
            try:
                optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
                scaler=torch.amp.GradScaler("cuda",enabled=config["amp"],init_scale=1024.)
                batch={k:v.unsqueeze(0).repeat(batch_size,*([1]*v.ndim)) for k,v in sample.items() if k!="index"}
                inputs=_device_batch(batch,device,config["channels_last"])
                torch.cuda.synchronize(device); before=time.perf_counter()
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    for micro in range(min(config["accumulation_steps"],2)):
                        with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]): logits=model(inputs["image"])
                        values=criterion(logits,inputs)
                        scaler.scale(values["loss"]/min(config["accumulation_steps"],2)).backward()
                        logits=values=None
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"])
                    scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize(device)
                if not optimizer.state: raise RuntimeError("AMP skipped all probe optimizer steps")
                free,total=torch.cuda.mem_get_info(device)
                peak=torch.cuda.max_memory_allocated(device); reserved=torch.cuda.max_memory_reserved(device)
                margin=max(384*1024**2,sum(p.numel()*p.element_size() for p in model.parameters())*2)
                report.update(fits=peak<total*config["memory_fraction"] and reserved<total*config["memory_fraction"] and free>margin,
                    allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                    peak_allocated_bytes=peak,peak_reserved_bytes=reserved,free_bytes=free,total_bytes=total,
                    ddp_margin_bytes=margin,seconds=time.perf_counter()-before,optimizer_state_allocated=True)
            except torch.cuda.OutOfMemoryError:
                report.update(oom=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            finally:
                model.zero_grad(set_to_none=True); optimizer=values=inputs=logits=scaler=None
                gc.collect(); torch.cuda.empty_cache()
            reports.append(report)
            print(f"[rank {rank}] GPU probe ({time.perf_counter()-before_probe:.1f}s total): {report}",flush=True)
    finally:
        model.load_state_dict(original); restore_rng(random_state); model.zero_grad(set_to_none=True)
    rank_reports=gather_objects(reports)
    feasible=[b for b in candidates if all(next(t for t in r if t["batch_size"]==b)["fits"] for r in rank_reports)]
    selected=(max(feasible) if requested=="auto" else (int(requested) if int(requested) in feasible else 1)) if feasible else None
    return selected,rank_reports


def train(config: Config, train_records: list, val_records: list, split_fingerprint: str):
    """Entry point for one CUDA process or a torchrun NCCL rank."""
    config.validate()
    rank=int(os.environ.get("RANK",0)); local_rank=int(os.environ.get("LOCAL_RANK",0))
    world=int(os.environ.get("WORLD_SIZE",1)); distributed=world>1
    startup=time.perf_counter()
    def progress(message):
        print(f"[rank {rank} +{time.perf_counter()-startup:.1f}s] {message}",flush=True)
    progress(f"Initializing CUDA device {local_rank}; world_size={world}")
    if not torch.cuda.is_available(): raise RuntimeError("Production training requires CUDA; synthetic unit tests run on CPU")
    if distributed and (os.name=="nt" or not dist.is_nccl_available()): raise RuntimeError("Multi-GPU training requires Linux CUDA/NCCL")
    device=torch.device("cuda",local_rank); torch.cuda.set_device(device)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    seed_everything(config.seed,config.deterministic)
    output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True)
    progress("CUDA ready")
    if distributed:
        progress("Initializing NCCL process group")
        dist.init_process_group("nccl",timeout=timedelta(minutes=30),device_id=device)
        progress("NCCL ready")
    try:
        previous=load_checkpoint(config.resume) if config.resume else None
        if previous:
            if previous["world_size"]!=world or previous["split_fingerprint"]!=split_fingerprint:
                raise ValueError("Resume world size or split fingerprint differs")
            for key in ("loss","normalization","augment","accumulation_steps","scheduler","separation_radius",
                        "seed","deterministic","centerline","learning_rate","weight_decay","grad_clip","amp"):
                if config.to_dict()[key]!=previous["config"][key]: raise ValueError(f"Resume changes {key}")
            config.model=previous["config"]["model"]
        if config.smoke:
            train_records=train_records[:max(world,config.smoke_records)]
            val_records=val_records[:config.smoke_records]; config.epochs=1
        train_set=FullImageDataset(train_records,True,config.seed,config.normalization,config.augment)
        val_set=FullImageDataset(val_records,False,config.seed,config.normalization)
        criterion=FilamentPQProxyLoss(**config.loss).to(device)
        widths=[config.model.get("init_features",config.model.get("base_channels",16))]
        if not previous and config.model.get("name","modified")=="modified":
            widths+=sorted({w for w in config.width_fallbacks if w<widths[0]},reverse=True)
        model=None; all_probes=[]
        for width in widths:
            progress(f"Building model width={width}; loading probe sample")
            if model is not None: del model; gc.collect(); torch.cuda.empty_cache()
            if config.model.get("name","modified")=="modified": config.model["init_features"]=width
            model=build_model(config.model).to(device)
            if config.channels_last: model.to(memory_format=torch.channels_last)
            microbatch,reports=probe_batch_sizes(model,criterion,train_set[0],config.to_dict(),device)
            all_probes.append(dict(width=width,ranks=reports))
            if microbatch is not None: break
        if microbatch is None: raise RuntimeError("No native-resolution training batch fits; reduce channel width or free GPU memory")
        config.microbatch=microbatch
        progress(f"Selected microbatch={microbatch} per GPU; effective batch={microbatch*world*config.accumulation_steps}")
        progress("Wrapping model with DDP" if distributed else "Setting up single-GPU training")
        model=DDP(model,device_ids=[local_rank],broadcast_buffers=False,gradient_as_bucket_view=True) if distributed else model
        sampler=DistributedSampler(train_set,num_replicas=world,rank=rank,seed=config.seed,drop_last=False)
        workers=config.workers//world+int(rank<config.workers%world)
        options=dict(batch_size=microbatch,pin_memory=True,worker_init_fn=seed_loader_worker)
        worker_options=dict(num_workers=workers)
        if workers: worker_options.update(prefetch_factor=config.prefetch_factor,persistent_workers=True,multiprocessing_context="spawn")
        generator=torch.Generator()
        phase_dataset=TrainValidationDataset(train_set,val_set)
        phase_sampler=PhaseSampler(sampler,list(range(rank,len(val_set),world)))
        loader=DataLoader(phase_dataset,sampler=phase_sampler,generator=generator,**options,**worker_options)
        progress(f"Loader ready: {workers} CPU workers on this rank; prefetch={config.prefetch_factor}")
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        scaler=torch.amp.GradScaler("cuda",enabled=config.amp)
        scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=2) if config.scheduler=="plateau"
            else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config.epochs) if config.scheduler=="cosine" else None)
        start=0; history=[]; early={}; best={}; elapsed_before=0.
        if previous:
            previous=resume_training(config.resume,model,optimizer,scheduler,scaler,device,rank)
            start=previous["epoch"]; history=previous["history"]; early=previous["early_stopping"]; best=previous["best_metrics"]
            elapsed_before=history[-1].get("total_training_seconds",0) if history else 0
            del previous
        hardware=gather_objects(dict(rank=rank,name=torch.cuda.get_device_name(device),cpu_workers=workers,
            vram_gib=torch.cuda.get_device_properties(device).total_memory/2**30))
        if rank==0:
            atomic_json(output/"effective_config.json",config.to_dict())
            versions={}
            for package in ("numpy","scipy","scikit-image","pandas","matplotlib","pillow","pycocotools","opencv-python","opencv-python-headless"):
                try: versions[package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError: pass
            atomic_json(output/"runtime_report.json",dict(hardware=hardware,probes=all_probes,world_size=world,
                torch_version=torch.__version__,cuda_version=torch.version.cuda,package_versions=versions))
        started=time.perf_counter()
        for epoch in range(start+1,config.epochs+1):
            progress(f"Starting epoch {epoch}/{config.epochs}")
            train_set.set_epoch(epoch); generator.manual_seed(config.seed+epoch+rank)
            sampler.set_epoch(epoch)
            phase_dataset.select(False); phase_sampler.validation=False
            torch.cuda.reset_peak_memory_stats(device)
            train_metrics=train_one_epoch(model,loader,criterion,optimizer,scaler,device,
                accumulation_steps=config.accumulation_steps,amp=config.amp,channels_last=config.channels_last,grad_clip=config.grad_clip,log_every=config.log_every)
            do_pq=should_compute_pq(epoch,config.epochs,config.pq_every)
            if rank==0:
                print(f"Epoch {epoch} training finished: {train_metrics['seconds']:.1f}s; "
                      f"starting validation (PQ={'on' if do_pq else 'off'})",flush=True)
            phase_dataset.select(True); phase_sampler.validation=True
            val_metrics,details=validate_one_epoch(model,loader,criterion,device,
                amp=config.amp,channels_last=config.channels_last,pq_records=val_records if do_pq else None,
                postprocess_config=config.postprocess,log_every=config.log_every)
            if scheduler:
                scheduler.step(val_metrics["loss"]) if config.scheduler=="plateau" else scheduler.step()
            improved=stop=False
            if do_pq:
                early,improved,stop=update_early_stopping(early,val_metrics["pq"],config.patience)
                if improved: best=dict(epoch=epoch,**val_metrics)
            memory=gather_objects(gpu_statistics(device))
            row=dict(epoch=epoch,**{f"train_{k}":v for k,v in train_metrics.items()},
                **{f"val_{k}":v for k,v in val_metrics.items()},learning_rate=optimizer.param_groups[0]["lr"],
                total_training_seconds=elapsed_before+time.perf_counter()-started,
                peak_allocated_gib=max(m["peak_allocated_gib"] for m in memory),
                peak_reserved_gib=max(m["peak_reserved_gib"] for m in memory),gpu_statistics=memory)
            history.append(row)
            random_states=gather_objects(rng_state())
            if rank==0:
                kwargs=dict(model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,epoch=epoch,
                    best_metrics=best,config=config.to_dict(),early_stopping=early,history=history,
                    rank_rng_states=random_states,split_fingerprint=split_fingerprint)
                if improved: save_checkpoint(output/"checkpoints/best.pt",**kwargs)
                save_checkpoint(output/"checkpoints/last.pt",**kwargs)
                log_training_metrics(output/"logs/history.csv",history)
                if do_pq: atomic_json(output/f"validation/epoch_{epoch:03d}.json",dict(metrics=val_metrics,records=details))
                print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, val={val_metrics['loss']:.4f}, "
                    f"PQ={val_metrics.get('pq','skipped')}, train={train_metrics['seconds']:.1f}s, "
                    f"validation={val_metrics['seconds']:.1f}s, "
                    f"postprocess={val_metrics.get('postprocess_seconds',0.):.1f}s, "
                    f"PQ scoring={val_metrics.get('pq_scoring_seconds',0.):.1f}s",flush=True)
            if distributed: dist.barrier()
            if stop: break
    except torch.cuda.OutOfMemoryError:
        atomic_json(output/f"oom_rank_{rank}.json",dict(rank=rank,microbatch=config.microbatch,model=config.model))
        raise
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()
