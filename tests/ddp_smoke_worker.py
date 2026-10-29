"""Two CPU/Gloo ranks: accumulation parity and unequal validation rank lengths.

Run directly, independently of pytest. This exercises DDP semantics without
pretending to validate Linux/NCCL or dual-T4 memory/throughput.
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler
from test_model_loss_training import ScalarModel,ScalarDataset,MSELoss
from filament.training import run_epoch
from filament.training import PhaseSampler
from filament.data import TrainValidationDataset,seed_loader_worker


def worker(rank,store_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=Path(store_path).as_uri())
    try:
        model=DistributedDataParallel(ScalarModel())
        dataset=ScalarDataset()
        sampler=DistributedSampler(dataset,num_replicas=2,rank=rank,shuffle=False,drop_last=False)
        phase_dataset=TrainValidationDataset(dataset,dataset)
        phase_sampler=PhaseSampler(sampler,list(range(rank,len(dataset),2)))
        loader=DataLoader(phase_dataset,batch_size=2,sampler=phase_sampler,num_workers=2,
            persistent_workers=True,prefetch_factor=2,multiprocessing_context="spawn",worker_init_fn=seed_loader_worker)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        result,_=run_epoch(model,loader,MSELoss(),'cpu',optimizer,accumulation_steps=2,amp=False,
            channels_last=False,grad_clip=1e6)
        reference=ScalarModel(); opt=torch.optim.SGD(reference.parameters(),lr=.01)
        # DistributedSampler pads the five-item dataset to [0,1,2,3,4,0].
        indices=[0,1,2,3,4,0]
        x=torch.tensor([[float(i+1)] for i in indices]); y=torch.tensor([[float(i)] for i in indices])
        loss=((reference(x)-y)**2).mean(); loss.backward(); opt.step()
        assert torch.allclose(model.module.weight,reference.weight,atol=1e-6)
        assert result['samples']==6
        phase_dataset.select(True); phase_sampler.validation=True
        metrics,_=run_epoch(model,loader,MSELoss(),'cpu',amp=False,channels_last=False)
        assert metrics['samples']==5
        if rank==0: print('PASS: two-rank Gloo accumulation matches reference; four persistent workers switch phases; validation counts each of five records once',flush=True)
        del loader
    finally: dist.destroy_process_group()


if __name__=='__main__':
    if not dist.is_gloo_available(): raise RuntimeError('Gloo unavailable')
    with tempfile.TemporaryDirectory(prefix='filament_ddp_') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'store'),),nprocs=2,join=True)
