import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class Synthesized_Dataset(Dataset):
    """synthesized dataset."""

    def __init__(self, csv_file, project_root, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            project_root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file, sep=",", dtype=np.float64)
        self.project_root = project_root
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = (
            torch.tensor(self.dataframe.loc[idx, "x1":"x2"].to_numpy())
            .reshape(-1, 2)
            .float()
        )
        y = (
            torch.tensor(self.dataframe.loc[idx, "target_y"])
            .reshape(-1, 1)
            .float()
        )
        sample = (x, y)

        if self.transform:
            sample = self.transform(sample)

        return sample