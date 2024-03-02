import torch
from syncbn import SyncBatchNorm
import torch.multiprocessing as mp
import pytest
import os
import torch.distributed as dist

def custom_loss(input):
    return torch.sum(input[:input.shape[0] // 2])

def proc_custom_loss(input):
    return torch.sum(input)


def init_process(rank, size, fn, batchnorm, x, global_bs, out_queue, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, batchnorm, x, global_bs, out_queue)


def run_batchnorm(rank, size, batchnorm, x, global_bs, out_queue):
    x.requires_grad_(True)
    output = batchnorm(x)
    # counting loss for each process
    local_bs = global_bs // size
    start_bs = local_bs * rank
    end_bs = min(start_bs + local_bs, global_bs // 2)
    loss = (start_bs < end_bs) * proc_custom_loss(output[:end_bs - start_bs])
    loss.backward()
    
    for _ in range(rank):
        dist.barrier()

    out_queue.put((output.detach(), x.grad.detach()))

    for _ in range(size - rank):
        dist.barrier()


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = torch.multiprocessing.get_context("spawn")

    original_batchnorm = torch.nn.BatchNorm1d(
        hid_dim, affine=False, eps=1e-5, momentum=0.1
    )
    custom_batchnorm = SyncBatchNorm(hid_dim, eps=1e-5, momentum=0.1)

    original_batchnorm.train()
    custom_batchnorm.train()

    x = torch.randn(batch_size, hid_dim, dtype=torch.float32)
    x_copy = x.clone()
    x_for_process = torch.chunk(x_copy, num_workers) # assume in our case they are the same size
    x.requires_grad_(True)
    

    original_output = original_batchnorm(x)
    original_loss = custom_loss(original_output)
    original_loss.backward()
    original_grad = x.grad

    out_queue = mp.Queue()
    processes = []
    for rank in range(num_workers):
        p = mp.Process(
            target=init_process,
            args=(
                rank,
                num_workers,
                run_batchnorm,
                custom_batchnorm,
                x_for_process[rank],
                batch_size,
                out_queue,
            ),
        )
        p.start()
        processes.append(p)

    custom_out = []
    custom_grad = []
    for _ in range(num_workers):
        item_out, item_grad = out_queue.get()
        custom_out.append(item_out)
        custom_grad.append(item_grad)

    custom_out = torch.cat(custom_out, 0)
    custom_grad = torch.cat(custom_grad, 0)

    for p in processes:
        p.join()

    assert torch.allclose(custom_out, original_output, atol=1e-3, rtol=0)
    assert torch.allclose(custom_grad, original_grad, atol=1e-3, rtol=0)
