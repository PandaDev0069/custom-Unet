"""Reproducible preparation and safe single/DDP process orchestration."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from .config import Config
from .data import (find_dataset_root,make_records,create_train_val_split,load_split_manifest,
    save_split_manifest,cache_training_targets,summarize_dataset,verify_no_leakage,TARGET_VERSION,discover_test_images)
from .io import atomic_json, fingerprint


def prepare_experiment(config: Config, statistics=True) -> dict:
    config.validate()
    root=find_dataset_root(config.data_root)
    config.data_root=str(root)
    output=Path(config.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    config.output_dir=str(output)
    records,integrity=make_records(root,config.image_size)
    integrity["test_source_images"]=len(discover_test_images(root,config.image_size))
    atomic_json(output/"data/integrity.json",integrity)
    split_path=output/"split_manifest.json"
    manifest=load_split_manifest(split_path,records) if split_path.exists() else create_train_val_split(
        records,config.val_fraction,config.seed,config.multi_annotator_train_only)
    if not split_path.exists(): save_split_manifest(split_path,manifest)
    # Annotation content as well as image identities bind the prepared runtime.
    selected=records
    if config.smoke:
        by_id={r["image_id"]:r for r in records}
        selected=[by_id[i] for i in manifest["train"][:config.smoke_records]+manifest["val"][:config.smoke_records]]
    data_identity=fingerprint(dict(split=manifest,annotations=integrity["annotation_sha256"],
        radius=config.separation_radius,centerline=config.centerline,
        selected_ids=[r["image_id"] for r in selected],smoke=config.smoke,target_version=TARGET_VERSION))
    if statistics:
        summary,rows=summarize_dataset(selected)
        atomic_json(output/"data/statistics.json",summary)
        pd.DataFrame(rows).to_csv(output/"data/statistics_per_record.csv",index=False)
    cached=cache_training_targets(selected,output/"targets",config.separation_radius,config.centerline)
    by_id={r["image_id"]:r for r in cached}
    train_records=[by_id[i] for i in manifest["train"] if i in by_id]
    val_records=[by_id[i] for i in manifest["val"] if i in by_id]
    verify_no_leakage(train_records,val_records)
    runtime=dict(config=config.to_dict(),train_records=train_records,val_records=val_records,
                 split_fingerprint=data_identity,smoke=config.smoke)
    atomic_json(output/"prepared.json",runtime)
    atomic_json(output/"config.json",config.to_dict())
    return runtime


def launch_training(runtime: dict, num_gpus=None) -> None:
    """Launch worker module; whole-job OOM fallback restarts all DDP ranks safely.

    Each failed epoch is replayed from last.pt. Smaller microbatch is automatic;
    resolution never changes. Architecture fallback is only allowed before any
    checkpoint exists. torchrun terminates peer ranks if one rank fails.
    """
    config=Config(**runtime["config"])
    available=torch.cuda.device_count()
    count=min(2,available) if num_gpus is None else int(num_gpus)
    if count<1 or count>available: raise ValueError("Requested GPU count unavailable")
    if count>1 and os.name=="nt": raise RuntimeError("NCCL DDP requires Linux; select one GPU on Windows")
    output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parent.parent
    environment=os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    environment["PYTHONPATH"]=str(source_root)+os.pathsep+environment.get("PYTHONPATH","")
    attempt=0
    while True:
        for stale in output.glob("oom_rank_*.json"): stale.unlink()
        run_file=output/"worker_runtime.json"
        runtime["config"]=config.to_dict(); atomic_json(run_file,runtime)
        command=[sys.executable,"-u"]
        if count>1: command += ["-m","torch.distributed.run","--standalone",f"--nproc_per_node={count}"]
        command += ["-m","filament.cli","worker","--runtime",str(run_file)]
        log=output/"logs"/f"training_attempt_{attempt}.log"; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w",encoding="utf8") as handle:
            process=subprocess.Popen(command,cwd=source_root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=os.name!="nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            try:
                for line in process.stdout:
                    print(line,end="",flush=True); handle.write(line); handle.flush()
                returncode=process.wait()
            except BaseException:
                if os.name!="nt":
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                else:
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
                process.wait(); raise
        if returncode==0: return
        signals=list(output.glob("oom_rank_*.json"))
        if not signals: raise RuntimeError(f"Training worker failed ({returncode}); inspect {log}")
        failures=[json.loads(p.read_text(encoding="utf8")) for p in signals]
        old_batch=min(int(r["microbatch"]) for r in failures)
        last=output/"checkpoints/last.pt"
        if old_batch>1:
            config.microbatch=max(1,old_batch//2)
            if last.exists(): config.resume=str(last)
        elif not last.exists() and not config.resume and config.model.get("name","modified")=="modified":
            old_width=failures[0]["model"]["init_features"]
            options=[w for w in config.width_fallbacks if w<old_width]
            if not options: raise RuntimeError(f"Native microbatch one exhausted width fallbacks; inspect {log}")
            config.model["init_features"]=max(options); config.microbatch=1
        else: raise RuntimeError(f"Native microbatch one cannot recover without changing a saved architecture; inspect {log}")
        attempt+=1
        print(f"Restarting all ranks after CUDA OOM with microbatch={config.microbatch}",flush=True)
