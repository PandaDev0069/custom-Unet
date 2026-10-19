"""python -m filament.cli prepare|train|worker|infer|smoke."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from .config import Config
from .io import atomic_json,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","train","worker","evaluate","infer","smoke"])
    parser.add_argument("--config"); parser.add_argument("--runtime")
    parser.add_argument("--gpus",type=int); parser.add_argument("--output",default="filament_outputs")
    args=parser.parse_args()
    if args.command=="worker":
        from .training import train
        runtime=json.loads(Path(args.runtime).read_text(encoding="utf8"))
        train(Config(**runtime["config"]),runtime["train_records"],runtime["val_records"],runtime["split_fingerprint"])
        return
    config=Config(**json.loads(Path(args.config).read_text(encoding="utf8"))) if args.config else Config(output_dir=args.output)
    if args.command=="smoke":
        config.smoke=True; config.epochs=1; config.microbatch=1
        config.model["init_features"]=8
    if args.command in ("prepare","train","smoke"):
        from .workflow import prepare_experiment,launch_training
        runtime=prepare_experiment(config)
        if args.command!="prepare": launch_training(runtime,args.gpus)
    elif args.command=="evaluate":
        from .training import load_checkpoint
        from .model import build_model
        from .inference import cache_validation_predictions,search_postprocessing
        from .visualization import plot_training,plot_distributions,plot_error_gallery
        import matplotlib.pyplot as plt
        output=Path(config.output_dir)
        runtime=json.loads((output/"prepared.json").read_text(encoding="utf8"))
        checkpoint=output/"checkpoints/best.pt"
        state=load_checkpoint(checkpoint); effective=Config(**state["config"])
        model=build_model(effective.model)
        model.load_state_dict(state["model"]); del state
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model=model.to(device)
        manifest=cache_validation_predictions(model,runtime["val_records"],output/"validation_predictions",device,
            dict(checkpoint_sha256=file_hash(checkpoint),normalization=effective.normalization,split_fingerprint=runtime["split_fingerprint"]),
            amp=effective.amp,channels_last=effective.channels_last,normalization=effective.normalization)
        grid=dict(method=["components","boundary","watershed"],foreground_threshold=[.4,.5,.6],min_area=[10,20])
        if effective.smoke: grid=dict(method=["components","boundary"],foreground_threshold=[.5])
        best,frame=search_postprocessing(runtime["val_records"],manifest,grid,output/"search",effective.postprocess)
        details=json.loads((output/"search/best_validation_details.json").read_text(encoding="utf8"))
        plt.close(plot_training(output/"logs/history.csv",output/"figures"))
        plt.close(plot_distributions(details,output/"figures/overlap_distributions.png"))
        plot_error_gallery(runtime["val_records"],manifest,details,best["postprocess"],output/"figures/errors")
        print(best)
    elif args.command=="infer":
        from .data import find_dataset_root,discover_test_images,read_grayscale
        from .training import load_checkpoint
        from .model import build_model
        from .inference import predict_full_image
        from .postprocess import extract_instances
        from .submission import generate_submission
        output=Path(config.output_dir)
        checkpoint=output/"checkpoints/best.pt"
        state=load_checkpoint(checkpoint); config=Config(**state["config"])
        if config.smoke: raise ValueError("A smoke checkpoint cannot generate a final test submission")
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=build_model(config.model).to(device); model.load_state_dict(state["model"])
        del state
        selection=json.loads((output/"search/best_postprocessing.json").read_text(encoding="utf8"))
        if selection["prediction_provenance"]["checkpoint_sha256"]!=file_hash(checkpoint):
            raise ValueError("Postprocessing selection belongs to a different checkpoint")
        post=selection["postprocess"]
        paths=discover_test_images(find_dataset_root(config.data_root))
        def predict(path):
            p=predict_full_image(model,read_grayscale(path),device,config.amp,config.channels_last,config.normalization)
            return extract_instances(p.astype(np.float16),post)[1]
        report=generate_submission(paths,predict,output/"submission.csv",output/"inference_journal",
            dict(checkpoint_sha256=file_hash(checkpoint),postprocess=post,normalization=config.normalization))
        print(report)


if __name__=="__main__": main()
