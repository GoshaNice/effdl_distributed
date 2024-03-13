import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from syncbn import SyncBatchNorm


def custom_loss(input):
    return torch.sum(input[: input.shape[0] // 2])


def proc_custom_loss(input):
    return torch.sum(input)


def init_process(rank, size, fn, global_bs, hid_dim, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group(backend, rank=rank, world_size=size)

    ## Choose your fighter
    batchnorm = nn.SyncBatchNorm(hid_dim, affine=False, device=f"cuda:{rank}") # switch to original
    # batchnorm = SyncBatchNorm(hid_dim, eps=1e-5, momentum=0.1)  # switch to custom
    batchnorm.train()
    fn(rank, size, batchnorm, hid_dim, global_bs)


def run_batchnorm(rank, size, batchnorm, hid_dim, global_bs):
    if rank == 0:
        print("---------------------")
        print(f"Hidden dim: {hid_dim}, Batch size: {global_bs}")
    times = []
    for i in range(10):
        x = torch.randn(
            global_bs,
            hid_dim,
            dtype=torch.float32,
            device=f"cuda:{rank}",
            requires_grad=True,
        )
        start_time = time.perf_counter()
        output = batchnorm(x)
        # counting loss for each process
        local_bs = global_bs // size
        start_bs = local_bs * rank
        end_bs = min(start_bs + local_bs, global_bs // 2)
        loss = (start_bs < end_bs) * proc_custom_loss(output[: end_bs - start_bs])
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start_time)

    if rank == 1:
        dist.barrier()
    print(f"[Rank {rank}]")
    print("Time:")
    print(f"Mean \t {np.mean(times)}")
    print(f"Median \t {np.median(times)}")
    print("Cuda memory summary:")
    print(torch.cuda.memory_summary(device=f"cuda:{rank}"))
    if rank == 0:
        dist.barrier()


def main(hid_dim, batch_size):
    ctx = torch.multiprocessing.get_context("spawn")
    num_workers = 2
    torch.multiprocessing.spawn(
        init_process,
        args=(
            num_workers,
            run_batchnorm,
            batch_size,
            hid_dim,
        ),
        join=True,
        nprocs=num_workers,
    )


if __name__ == "__main__":
    for hid_dim in [128, 256, 512, 1024]:
        for batch_size in [32, 64]:
            main(hid_dim, batch_size)
