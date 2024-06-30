import torch
import torch.nn as nn


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        if size > self.max_size:
            size = self.max_size
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        if self.current_size == 0:
            return torch.nan
        else:
            return self.mean.squeeze(0).cpu()
