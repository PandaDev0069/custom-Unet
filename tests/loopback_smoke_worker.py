"""Run via torchrun to verify its single-host loopback rendezvous on CPU."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist


if __name__ == "__main__":
    torch.set_num_threads(1)
    assert os.environ["MASTER_ADDR"] == "127.0.0.1", os.environ["MASTER_ADDR"]
    if os.name=="nt":
        # Test-only compatibility: this Windows wheel's TCPStore has no libuv.
        # The elastic agent already owns the server; both ranks are clients.
        store=dist.TCPStore(os.environ["MASTER_ADDR"],int(os.environ["MASTER_PORT"]),
            is_master=False,timeout=timedelta(seconds=60),use_libuv=False)
        dist.init_process_group("gloo",store=store,rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),timeout=timedelta(seconds=60))
    else:
        dist.init_process_group("gloo",timeout=timedelta(seconds=60))
    try:
        value=torch.tensor(float(dist.get_rank()+1))
        dist.all_reduce(value)
        assert value.item()==3 and dist.get_world_size()==2
        if dist.get_rank()==0:
            print("PASS: two torchrun ranks rendezvous at 127.0.0.1 and complete all-reduce",flush=True)
    finally:
        dist.destroy_process_group()
