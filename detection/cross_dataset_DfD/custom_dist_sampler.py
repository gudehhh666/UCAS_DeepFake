from torch.utils.data import DistributedSampler
import torch

class CustomDSampler(DistributedSampler) :
    def __init__(self, dataset, num_replicas = 1, rank = 1, shuffle = True, seed = 0, n_sample = -1, p0 = 0.5):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.p0 = p0
    
        self.num_samples = n_sample
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
 

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            raise NotImplementedError

        classes = [[], []]
        for index in indices :
            cls = self.dataset[index][1]
            classes[cls].append(index)

        pool = []

        sample_indices = []
        while len(sample_indices) < self.num_replicas*self.num_samples :
            if torch.rand(1, generator=g).item() < self.p0:
                cls_idx = 0
            else:
                cls_idx = 1

            idx = classes[cls_idx][0]
            classes[cls_idx] = classes[cls_idx][1: ] + classes[cls_idx][: 1]
            sample_indices.append(idx)

        # subsample
        sample_indices = sample_indices[self.rank:self.total_size:self.num_replicas]
        assert len(sample_indices) == self.num_samples
        return iter(sample_indices)
