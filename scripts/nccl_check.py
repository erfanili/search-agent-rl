"""
One NCCL broadcast between the visible GPUs, with a short timeout. Run by scripts/nccl_check.sh.
Prints OK and the time taken, or fails within ~60 s if the transfer hangs.
"""

import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
    x = torch.full((15_000_000,), float(rank == 0), device="cuda")  # ~60 MB, like verl's first broadcast
    torch.cuda.synchronize()
    t0 = time.time()
    dist.broadcast(x, src=0)
    torch.cuda.synchronize()
    ok = bool(torch.all(x == 1).item())
    if rank == 0:
        print(f"broadcast across {world} GPUs: {'OK' if ok else 'WRONG DATA'} in {time.time() - t0:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    world = torch.cuda.device_count()
    if world < 2:
        raise SystemExit("need at least 2 visible GPUs")
    mp.spawn(worker, args=(world,), nprocs=world)
