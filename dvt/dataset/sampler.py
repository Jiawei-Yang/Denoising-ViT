import math

import numpy as np
from torch.utils.data import Sampler


class InfiniteSampler(Sampler):
    """Sampler that yields indices infinitely."""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            for i in range(len(self.data_source)):
                yield i


class DistributedInfiniteSampler(Sampler):
    """Sampler that yields indices infinitely, adapted for distributed training, with shuffling."""

    def __init__(self, data_source, num_replicas=None, rank=None):
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        indices = list(range(len(self.data_source)))
        # Divide indices among replicas
        indices = [indices[i :: self.num_replicas] for i in range(self.num_replicas)]
        # Shuffle only the subset for this replica using the seeded RNG
        rng.shuffle(indices[self.rank])

        while True:
            yield from indices[self.rank]

    def __len__(self):
        return self.num_samples
