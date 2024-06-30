import torch


def grad_norm(params) -> torch.Tensor:
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad**2)
    return torch.sqrt(grad_norm)


def print_gpu_memory(device=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert bytes to MB
        cached = torch.cuda.memory_reserved(device) / 1024**2  # Convert bytes to MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"Device {device}:")
        print(f"Allocated Memory: {allocated:.2f} MB")
        print(f"Cached Memory: {cached:.2f} MB")
        print(f"Reserved Memory: {reserved:.2f} MB")
    else:
        print("CUDA is not available.")
