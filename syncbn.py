import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        # Input (C,L)

        input_sum = input.sum(0)  # (L)
        input_squared_sum = (input**2).sum(0)  # (L)
        input_num_elem = torch.tensor([input.shape[0]])  # (1)
        input_info = torch.cat(
            [input_sum, input_squared_sum, input_num_elem], 0
        )  # (2L+1)

        dist.all_reduce(input_info, op=dist.ReduceOp.SUM)  # (2L+1)
        sum_elem = input_info[: input.shape[1]]  # (L)
        squared_sum_elem = input_info[input.shape[1] : 2 * input.shape[1]]  # (L)
        num_elem = input_info[2 * input.shape[1] : 2 * input.shape[1] + 1]  # (1)
        mean = sum_elem / num_elem  # (L)
        var = squared_sum_elem / num_elem - mean**2  # (L)
        std = torch.sqrt(var + eps)  # (L)
        normalized = (input - mean) / std  # (C, L)

        # ctx.save_for_backward(input, mean, std, num_elem)
        ctx.save_for_backward(num_elem, std, normalized)
        running_mean.data = (running_mean * momentum + mean * (1 - momentum)).data
        running_std.data = (running_std * momentum + std * (1 - momentum)).data
        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        # Credits to : https://i.stack.imgur.com/iWINr.png
        num_elem, std, normalized = ctx.saved_tensors
        input_grad_var = -(grad_output * normalized / (2 * std**2)).sum(0)
        input_sum_grad = grad_output.sum(0)
        info_grad = torch.cat([input_grad_var, input_sum_grad], 0)

        dist.all_reduce(info_grad, op=dist.ReduceOp.SUM)
        grad_var = info_grad[: normalized.shape[1]]
        sum_grad = info_grad[normalized.shape[1] : 2 * normalized.shape[1]]

        grad_mean = -sum_grad / std
        grad_input = (
            grad_output / std
            + grad_var * 2 * normalized * std / num_elem
            + grad_mean / num_elem
        )

        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        if self.training:
            output = sync_batch_norm.apply(
                input, self.running_mean, self.running_std, self.eps, self.momentum
            )
        else:
            output = (input - self.running_mean) / self.running_std

        return output
