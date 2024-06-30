from typing import Tuple
import torch
from torch import Tensor


class RunningMeanStd(object):
    def __init__(
        self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cuda:0"
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.shape = shape
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def to(self, device):
        rms = RunningMeanStd(device=device)
        rms.mean = self.mean.to(device).clone()
        rms.var = self.var.to(device).clone()
        rms.count = self.count
        return rms

    @torch.no_grad()
    def update(self, arr: Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        if batch_mean.shape != ():
            assert batch_mean.shape == self.mean.shape
            assert batch_var.shape == self.var.shape
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: Tensor, batch_var: Tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: Tensor, un_norm=False) -> Tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
            result = result.clamp(-5.0, 5.0)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean
        return result
