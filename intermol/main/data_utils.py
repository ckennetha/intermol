import os
import torch
import numpy as np
import polars as pl
import pytorch_lightning as ptl

from torch.utils.data import Dataset, DataLoader

class MolDataset(Dataset):
    def __init__(self, df: pl.DataFrame, col_name: str):
        super().__init__()
        self.df = df.select(["id", col_name])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.row(idx)
        return {'id': row[0], 'smi': row[1]}

class MolDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        col_name: str,
        batch_size: int,
        train_size: float,
        seed: int = None
    ):
        super().__init__()

        self.data_path = data_path
        self.col_name = col_name
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed

    def setup(self, stage = None):
        df = pl.read_parquet(self.data_path)
        rng = np.random.default_rng(self.seed)

        is_train = pl.Series(
            rng.choice(
                [True, False],
                size=len(df),
                replace=True,
                p=[self.train_size, 1 - self.train_size]
            )
        )
        self.df_train, df_val_test = df.filter(is_train), df.filter(~is_train)

        is_val = pl.Series(
            rng.choice(
                [True, False],
                size=len(df_val_test),
                replace=True,
                p=[0.5, 0.5]
                )
        )
        self.df_val = df_val_test.filter(is_val)
        self.df_test = df_val_test.filter(~is_val)

    def _make_dataloader(self, df: pl.DataFrame, shuffle: bool):
        num_workers = min(os.cpu_count(), 4)
        return DataLoader(
            MolDataset(df, self.col_name),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def train_dataloader(self):
        return self._make_dataloader(self.df_train, True)

    def val_dataloader(self):
        return self._make_dataloader(self.df_val, False)

    def test_dataloader(self):
        return self._make_dataloader(self.df_test, False)
