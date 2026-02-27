import numpy as np
import polars as pl
import pytorch_lightning as ptl

from torch.utils.data import Dataset, DataLoader

class MolDataset(Dataset):
    def __init__(self, df: pl.DataFrame, col_sele: str):
        super().__init__()
        self.df = df
        self.col_sele = col_sele

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        return {
            'id': row['id'],
            'smi': row[self.col_sele]
        }

class MolDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        data_pth: str,
        col_sele: str,
        batch_size: int,
        train_size: float = 0.8,
        num_workers: int = 1,
        seed: int = None
    ):
        super().__init__()

        self.data_pth = data_pth
        self.col_sele = col_sele
        self.batch_size = batch_size
        self.train_size = train_size

        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage = None):
        df = pl.read_parquet(self.data_pth)

        # train-val-test split
        if self.seed:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()

        is_train = pl.Series(
            rng.choice(
                [True, False],
                size=len(df),
                replace=True,
                p=[self.train_size, 1 - self.train_size]
            )
        )
        self.train_df, val_test_df = df.filter(is_train), df.filter(~is_train)

        is_val = pl.Series(
        rng.choice(
            [True, False],
            size=len(val_test_df),
            replace=True,
            p=[0.5, 0.5]
            )
        )
        self.val_df = val_test_df.filter(is_val)
        self.test_df = val_test_df.filter(~is_val)

    def train_dataloader(self):
        return DataLoader(
            MolDataset(self.train_df, self.col_sele),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            MolDataset(self.val_df, self.col_sele),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            MolDataset(self.test_df, self.col_sele),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
