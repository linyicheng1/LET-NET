import torch


class ConcatDatasets(torch.utils.data.Dataset):
    """
    Concat any kinds of torch datasets
    """

    def __init__(self, *datasets):
        """
        :param datasets: list of datasets
        """
        self.datasets = datasets
        self.offsets = []
        self.length = 0
        for d in self.datasets:
            self.length += len(d)
            self.offsets.append(self.length)

    def __getitem__(self, item):
        dataset_idx = 0
        item_idx = 0
        for i, offset in enumerate(self.offsets):
            if item >= offset:
                continue
            dataset_idx = i
            if i > 0:
                item_idx = item - self.offsets[i - 1]
            else:
                item_idx = item
            break
        return self.datasets[dataset_idx][item_idx]

    def __len__(self):
        return self.length
