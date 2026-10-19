"""Readable complete-disk and native-pixel detail plots saved as PNG artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import read_grayscale,cached_instances
from .postprocess import extract_instances


def _save(fig,path):
    if path:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=180,bbox_inches="tight")
    return fig


def _overlay(ax,image,masks,title):
    rgb=np.repeat(image[...,None],3,axis=-1)
    colors=plt.get_cmap("turbo")
    for i,mask in enumerate(masks):
        color=np.asarray(colors((i*.61803398875)%1))[:3]*255
        rgb[mask]=(rgb[mask].astype(np.float32)*.4+color*.6).astype(np.uint8)
    # One composited RGB layer avoids retaining N full-resolution RGBA layers.
    ax.imshow(rgb,interpolation="nearest")
    ax.set_title(title); ax.axis("off")


def _zoom_box(masks,shape,size=256):
    union=np.zeros(shape,bool)
    for mask in masks: union|=mask
    ys,xs=np.where(union)
    cy,cx=(int(np.median(ys)),int(np.median(xs))) if len(ys) else (shape[0]//2,shape[1]//2)
    if len(ys):
        nearest=int(np.argmin((ys-cy)**2+(xs-cx)**2))
        cy,cx=int(ys[nearest]),int(xs[nearest])
    y=max(0,min(cy-size//2,max(0,shape[0]-size)))
    x=max(0,min(cx-size//2,max(0,shape[1]-size)))
    return slice(y,y+size),slice(x,x+size)


def plot_dataset_inspection(record,path=None):
    image=read_grayscale(record["image_path"]); masks=cached_instances(record)
    with np.load(record["target_path"]) as target:
        fg,sep,valid=(target[k] for k in ("foreground","separation","separation_valid"))
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout="constrained")
    _overlay(axes[0,0],image,[],"Original H-alpha")
    _overlay(axes[0,1],image,masks,"Independent annotations (overlaps retained)")
    axes[0,2].imshow(fg,cmap="gray"); axes[0,2].set_title("Foreground union")
    axes[0,3].imshow(sep,cmap="magma",vmin=0,vmax=1); axes[0,3].set_title("Inter-instance separation")
    zoom=_zoom_box(masks,image.shape)
    _overlay(axes[1,0],image[zoom],[m[zoom] for m in masks],"Native pixels: annotation detail")
    for ax,data,title in zip(axes[1,1:],(fg[zoom],sep[zoom],~valid[zoom]),("Foreground detail","Separation detail","Excluded overlap neighborhood")):
        ax.imshow(data,cmap="gray",interpolation="nearest"); ax.set_title(title)
    for ax in axes.flat: ax.axis("off")
    fig.suptitle(record["file_name"],fontsize=15)
    return _save(fig,path)


def plot_training(history_path,output_dir):
    frame=pd.read_csv(history_path)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained")
    for split in ("train","val"): axes[0,0].plot(frame.epoch,frame[f"{split}_loss"],"o-",markersize=3,label=split)
    axes[0,0].set_title("Composite loss"); axes[0,0].legend()
    for name in ("bce","dice","boundary","cldice"):
        axes[0,1].plot(frame.epoch,frame[f"train_{name}"],"o-",markersize=3,label=name)
        axes[0,2].plot(frame.epoch,frame[f"val_{name}"],"o-",markersize=3,label=name)
    axes[0,1].set_title("Training terms"); axes[0,2].set_title("Validation terms")
    axes[0,1].legend(); axes[0,2].legend()
    axes[1,0].plot(frame.epoch,frame.learning_rate,"o-",markersize=3); axes[1,0].set_title("Learning rate")
    for col in ("peak_allocated_gib","peak_reserved_gib"): axes[1,1].plot(frame.epoch,frame[col],"o-",markersize=3,label=col)
    axes[1,1].set_title("GPU memory (GiB)"); axes[1,1].legend(fontsize=8)
    if "val_pq" in frame: axes[1,2].plot(frame.epoch,frame.val_pq,"o-")
    axes[1,2].set_title("Instance PQ used for model selection")
    for ax in axes.flat: ax.set_xlabel("Epoch"); ax.grid(alpha=.25)
    return _save(fig,Path(output_dir)/"training.png")


def plot_prediction(record,probabilities,postprocess_config,path=None,title=None):
    image=read_grayscale(record["image_path"]); gt=cached_instances(record)
    labels,pred=extract_instances(probabilities,postprocess_config)
    fig,axes=plt.subplots(2,6,figsize=(23,8),layout="constrained")
    zoom=_zoom_box(gt or pred,image.shape)
    for row,region in enumerate(((slice(None),slice(None)),zoom)):
        _overlay(axes[row,0],image[region],[],"Original" if row==0 else "Native-pixel detail")
        for col,channel,name in ((1,0,"Foreground probability"),(2,1,"Separation probability")):
            axes[row,col].imshow(probabilities[channel][region],cmap="magma",vmin=0,vmax=1,interpolation="nearest")
            axes[row,col].set_title(name)
        axes[row,3].imshow(labels[region]>0,cmap="gray",interpolation="nearest"); axes[row,3].set_title("Retained foreground")
        _overlay(axes[row,4],image[region],(m[region] for m in pred),f"Predicted instances ({len(pred)})")
        _overlay(axes[row,5],image[region],[m[region] for m in gt],f"GT instances ({len(gt)})")
    for ax in axes.flat: ax.axis("off")
    fig.suptitle(title or record["file_name"])
    return _save(fig,path)


def plot_distributions(details,path=None):
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    for ax,key,title in zip(axes[:2],("iou_distribution","dice_distribution"),("Positive-pair IoU","Positive-pair Dice")):
        values=[x for r in details for x in r[key]]
        weights=np.ones(len(values))/max(len(values),1)
        ax.hist(values,bins=np.linspace(0,1,51),weights=weights,color="navy",alpha=.7)
        ax.set(title=title,xlim=(0,1),ylabel="Probability",xlabel="Score")
    for key,label,offset in (("gt_overlap_degrees","GT 1:n",.2),("pred_overlap_degrees","Pred n:1",-.2)):
        degrees=[d for r in details for d in r[key]]
        bins=np.bincount(degrees) if degrees else np.zeros(1)
        axes[2].bar(np.arange(len(bins))+offset,bins,width=.4,label=label)
    axes[2].set(title="Any-overlap relations",xlabel="Overlap degree",ylabel="Instances"); axes[2].legend()
    return _save(fig,path)


def plot_error_gallery(records,manifest,details,postprocess_config,output_dir):
    if not details: return []
    by_id={r["image_id"]:r for r in records}
    choices={"best":max(details,key=lambda r:r["pq"]),"worst":min(details,key=lambda r:r["pq"])}
    for name,key in (("false_positives","fp"),("missed","fn"),("fragmentation","fragmented_gt"),("overmerging","merged_predictions")):
        example=max(details,key=lambda r:r[key])
        if example[key]>0: choices[name]=example
    outputs=[]
    for name,result in choices.items():
        record=by_id[result["image_id"]]
        probabilities=np.load(manifest["entries"][record["file_name"]]["path"],mmap_mode="r")
        path=Path(output_dir)/f"{name}.png"
        fig=plot_prediction(record,probabilities,postprocess_config,path,
            f"{name}: PQ={result['pq']:.3f}, FP={result['fp']}, FN={result['fn']}")
        plt.close(fig); outputs.append(str(path))
    return outputs
