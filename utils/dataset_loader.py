import torch
from torch.utils.data import Dataset


class RadioMLDataset(Dataset):

    def __init__(self, dataset_path, split="train"):

        data = torch.load(dataset_path, weights_only=False)

        if split == "train":
            self.X = data["X_train"]
            self.y = data["y_train"]

        elif split == "valid":
            self.X = data["X_valid"]
            self.y = data["y_valid"]

        elif split == "test":
            self.X = data["X_test"]
            self.y = data["y_test"]

        else:
            raise ValueError("Invalid split")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

       
        x = self.X[index].float()
        y = self.y[index].long()

        return x, y