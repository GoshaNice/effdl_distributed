import torch
from syncbn import SyncBatchNorm
import torch.multiprocessing as mp
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

def main(hid_dim, batch_size):
    ctx = torch.multiprocessing.get_context("spawn")
    custom_batchnorm = SyncBatchNorm(hid_dim, eps=1e-5, momentum=0.1)

    custom_batchnorm.train()
    num_workers = 2
    x = torch.randn(batch_size, hid_dim, dtype=torch.float32)
    x_for_process = torch.chunk(x, num_workers)


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
        
if __name__=="__main__":
    for hid_dim in [128, 256, 512, 1024]:
        for batch_size in [32, 64]:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as prof:
                with torch.profiler.record_function(f"main_for_{hid_dim}_{batch_size}"):
                    main(hid_dim, batch_size)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
                
            
