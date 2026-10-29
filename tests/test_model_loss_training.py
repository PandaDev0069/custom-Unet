import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,Dataset
from unet import ModifiedUNet
from filament.losses import FilamentPQProxyLoss,instance_boundaries
from filament.training import (run_epoch,save_checkpoint,load_checkpoint,resume_training,update_early_stopping,rng_state,probe_batch_sizes,_device_batch,should_compute_pq)
from filament.config import Config
from filament.model import build_model


@pytest.mark.parametrize("shape",[(32,48),(33,35)])
def test_model_shapes_and_checkpoint_backward(shape):
    model=ModifiedUNet(init_features=2)
    image=torch.randn(1,1,*shape)
    output=model(image)
    assert output.shape==(1,2,*shape)
    target=dict(foreground=torch.zeros_like(output[:,:1]),separation=torch.zeros_like(output[:,:1]),separation_valid=torch.ones_like(output[:,:1]))
    result=FilamentPQProxyLoss(cldice_weight=.1)(output,target)
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert result["loss"].dtype==torch.float32


def test_exterior_and_ambiguous_loss():
    ids=torch.zeros(1,16,16,dtype=torch.long); ids[:,4:12,4:8]=1
    assert not instance_boundaries(ids).any()
    ids[:,4:12,8:12]=2
    assert instance_boundaries(ids).sum()==16
    logits=torch.randn(1,2,16,16,requires_grad=True)
    target=dict(foreground=torch.ones(1,1,16,16),separation=torch.ones(1,1,16,16),separation_valid=torch.zeros(1,1,16,16))
    values=FilamentPQProxyLoss()(logits,target); values["loss"].backward()
    assert values["boundary"]==0 and logits.grad[:,1].abs().sum()==0
    assert logits.grad[:,0].abs().sum()>0


def test_experimental_model_is_explicit():
    assert isinstance(build_model(dict(init_features=2)),ModifiedUNet)
    model=build_model(dict(name="full_image",in_channels=1,out_channels=2,
        base_channels=2,depth=3,max_channels=16,checkpoint_blocks=False))
    with torch.inference_mode():
        assert model(torch.zeros(1,1,33,35)).shape==(1,2,33,35)


class ScalarDataset(Dataset):
    def __len__(self): return 5
    def __getitem__(self,i): return dict(image=torch.tensor([float(i+1)]),target=torch.tensor([float(i)]),index=i)


class ScalarModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.25))
    def forward(self,x): return x*self.weight


class MSELoss(torch.nn.Module):
    def forward(self,logits,targets):
        loss=((logits-targets["target"])**2).mean()
        return dict(loss=loss,bce=loss.detach(),dice=loss.detach(),boundary=loss.detach(),cldice=loss.detach())


@pytest.mark.parametrize("epochs,every,expected",[(1,2,[1]),(5,2,[1,2,4,5]),(6,5,[1,5,6]),(4,1,[1,2,3,4])])
def test_pq_cadence_includes_first_and_final(epochs,every,expected):
    assert [epoch for epoch in range(1,epochs+1) if should_compute_pq(epoch,epochs,every)]==expected


def test_exact_short_accumulation_and_validation(tmp_path):
    model=ScalarModel(); reference=ScalarModel(); loader=DataLoader(ScalarDataset(),batch_size=2)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    reference_opt=torch.optim.SGD(reference.parameters(),lr=.01)
    criterion=MSELoss()
    for indices in ([0,1,2,3],[4]):
        x=torch.tensor([[float(i+1)] for i in indices]); y=torch.tensor([[float(i)] for i in indices])
        loss=((reference(x)-y)**2).mean(); loss.backward(); reference_opt.step(); reference_opt.zero_grad()
    metrics,_=run_epoch(model,loader,criterion,"cpu",opt,accumulation_steps=2,amp=False,channels_last=False,grad_clip=1e6)
    assert torch.allclose(model.weight,reference.weight,atol=1e-6)
    before=model.weight.detach().clone()
    run_epoch(model,loader,criterion,"cpu",channels_last=False)
    assert torch.equal(model.weight,before) and not model.training
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,1)
    path=tmp_path/"last.pt"
    save_checkpoint(path,model,opt,scheduler,scaler,1,dict(pq=.3),{},dict(best_pq=.3),[],[rng_state()],"split")
    next_random=torch.rand(4)
    with torch.no_grad(): model.weight.fill_(9)
    state=load_checkpoint(path,model,opt,scheduler,scaler)
    assert torch.equal(model.weight,before) and state["epoch"]==1
    resume_training(path,model,opt,scheduler,scaler,"cpu")
    assert torch.equal(torch.rand(4),next_random)
    early,improved,stop=update_early_stopping({},.3,2)
    assert improved and not stop
    early,_,_=update_early_stopping(early,.2,2)
    assert update_early_stopping(early,.2,2)[2]


@pytest.mark.gpu
def test_gpu_amp_training_path():
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model=ModifiedUNet(init_features=2).cuda()
    image=torch.randn(1,1,33,35,device="cuda")
    with torch.autocast("cuda",dtype=torch.float16): output=model(image)
    targets={k:torch.zeros(1,1,33,35,device="cuda") for k in ("foreground","separation")}
    targets["separation_valid"]=torch.ones_like(targets["foreground"])
    scaler=torch.amp.GradScaler("cuda",init_scale=128)
    optimizer=torch.optim.AdamW(model.parameters())
    loss=FilamentPQProxyLoss().cuda()(output,targets)["loss"]
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    assert optimizer.state and torch.isfinite(loss)


@pytest.mark.parametrize("batch_size",[1,2])
def test_grayscale_batch_explicit_channels_last_stride(batch_size):
    image=torch.randn(batch_size,1,32,32)
    batch=dict(image=image,foreground=torch.zeros_like(image),index=torch.arange(batch_size))
    converted=_device_batch(batch,"cpu",True)
    assert converted["image"].stride()==(1024,1,32,1)
    assert torch.equal(converted["image"],image) and "index" not in converted
    assert converted["foreground"].stride()==image.stride()


@pytest.mark.gpu
def test_cuda_grayscale_amp_gradient_layout_with_accumulation():
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model=ModifiedUNet(init_features=2,checkpoint_blocks=True).cuda().to(memory_format=torch.channels_last)
    criterion=FilamentPQProxyLoss().cuda()
    optimizer=torch.optim.AdamW(model.parameters())
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    # Full batch and short final batch: retain no_sync-style accumulated grads.
    # This reproduced [9,9,3,1] vs [9,1,3,1] on CUDA before the input fix.
    for batch_size in (2,2,1):
        image=torch.randn(batch_size,1,32,32)
        batch=dict(image=image,foreground=torch.zeros_like(image),separation=torch.zeros_like(image),
                   separation_valid=torch.ones_like(image))
        inputs=_device_batch(batch,"cuda",True)
        with torch.autocast("cuda",dtype=torch.float16): logits=model(inputs["image"])
        scaler.scale(criterion(logits,inputs)["loss"]/3).backward()
        assert all(p.grad is not None and p.grad.stride()==p.stride() for p in model.parameters())
    scaler.step(optimizer); scaler.update()
    assert optimizer.state


@pytest.mark.native
def test_native_gpu_training_path():
    if os.environ.get("FILAMENT_NATIVE_SMOKE")!="1" or not torch.cuda.is_available(): pytest.skip("Set FILAMENT_NATIVE_SMOKE=1 with CUDA")
    model=ModifiedUNet(init_features=8).cuda().to(memory_format=torch.channels_last)
    sample=dict(image=torch.rand(1,2048,2048),foreground=torch.zeros(1,2048,2048),
        separation=torch.zeros(1,2048,2048),separation_valid=torch.ones(1,2048,2048))
    sample["foreground"][:,400:1600,900:910]=1
    config=Config().to_dict(); config.update(microbatch=1,accumulation_steps=2)
    selected,reports=probe_batch_sizes(model,FilamentPQProxyLoss().cuda(),sample,config,"cuda")
    assert selected==1 and reports[0][0]["optimizer_state_allocated"]
