from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, X, labels, device):
        self.X = X  # Assumed to be a torch.Tensor
        self.labels = labels
        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Load the data
        datapoint_X = self.X[idx].to(self.device)  # (n_features, )
        datapoint_label = self.labels[idx].to(self.device)  # (2, )
        batch_idx = idx
        return batch_idx, datapoint_X, datapoint_label
