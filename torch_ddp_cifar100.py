import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import DistributedOptimizer
from torchvision.datasets import CIFAR100
from syncbn import SyncBatchNorm

torch.set_num_threads(1)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = nn.SyncBatchNorm(
            128, affine=False
        )  # to be replaced with SyncBatchNorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(0)
    if rank == 0:
        start_time = time.perf_counter()

    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        ),
        download=True,
    )
    # where's the validation dataset?
    loader = DataLoader(
        dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=64
    )
    
    device = torch.device(f"cuda:{rank}") # replace with "cuda" afterwards
    model = Net().to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)
    total_accum_steps = 2
    remaining_accum_steps = 2

    for _ in range(10):
        epoch_loss = torch.zeros((1,), device=device)
        epoch_acc = torch.zeros((1,), device=device)

        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target) / total_accum_steps
            acc = (output.argmax(dim=1) == target).float().mean()
            epoch_acc += acc.detach()
            epoch_loss += loss.detach()
            loss.backward()
            
            remaining_accum_steps -= 1
            if remaining_accum_steps == 0:
                average_gradients(model)
                optimizer.step()
                remaining_accum_steps = total_accum_steps

        print(
            f"Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}, acc: {epoch_acc / num_batches}"
        )
        epoch_loss = 0
        # where's the validation loop?
    if rank == 1:
        dist.barrier()
    if rank == 0:
        print(f"Total time: {time.perf_counter()}")
        print(torch.cuda.memory_summary(device=f"cuda:{rank}"))
        dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(
        local_rank, fn=run_training, backend="nccl"
    )  # replace with "nccl" when testing on several GPUs
