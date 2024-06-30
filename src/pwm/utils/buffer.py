import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, buffer_size, batch_size, horizon, device, terminate=False):
        self._device = device
        self._capacity = buffer_size
        self._horizon = horizon
        self._sampler = SliceSampler(
            num_slices=batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
        )
        self._batch_size = batch_size * (horizon + 1)
        self._num_eps = 0
        self.terminate = terminate

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=12,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes/1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = "cuda" if 2.5 * total_bytes < mem_free else "cpu"
        # storage_device = "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td["obs"]
        action = td["action"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        if self.terminate:
            term = td["term"][1:].unsqueeze(-1)
            # task = td["task"][0] if "task" in td.keys() else None
            return self._to_device(obs, action, reward, term)
        else:
            return self._to_device(obs, action, reward)

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = (
            torch.ones_like(td["reward"].squeeze(), dtype=torch.int32) * self._num_eps
        )
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def add_batch(self, td):
        """Add a batch of episodes to the buffer."""
        num_eps = td["reward"].shape[0]
        ep_len = td["reward"].shape[1]
        max_eps = self._capacity // ep_len
        eps_that_fit = min(num_eps, max_eps)
        print(f"Can fit {eps_that_fit} episodes into buffer")
        td = td[torch.randint(0, num_eps, (eps_that_fit,))]
        episodes = torch.ones_like(td["reward"], dtype=torch.int32) * torch.arange(
            0, eps_that_fit, dtype=torch.int32
        ).view((-1, 1))
        td["episode"] = episodes
        # if "term" not in td.keys():
        #     td["term"] = torch.zeros_like(td["reward"], dtype=torch.bool)
        td = td.flatten()  # faltten to easy ading
        if self._num_eps == 0:
            self._buffer = self._init(td[0 : ep_len + 1])
        self._buffer.extend(td)
        self._num_eps += num_eps
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self._horizon + 1).permute(1, 0)
        return self._prepare_batch(td)

    def save(self, filepath):
        self._buffer.dumps(filepath)

    def load(self, filepath):
        if self._num_eps == 0:
            self._buffer = self._reserve_buffer(
                LazyTensorStorage(self._capacity, device=self._device)
            )
        self._buffer.loads(filepath)
