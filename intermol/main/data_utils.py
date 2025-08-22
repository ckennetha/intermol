import multiprocessing as mp
import polars as pl
import pytorch_lightning as ptl
from torch.utils.data import Dataset, DataLoader
from utils import train_val_test_split

class MolDataset(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        col_sele: str
    ):
        super().__init__()
        self.data = data
        self.col_sele = col_sele

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)

        return {
            'smi': row[self.col_sele],
            'id': row['id']
        }

class MolDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        data_pth: str,
        col_sele: str,
        batch_size: int,
        train_size: float=0.8,
        num_workers: int=mp.cpu_count() - 1,
        seed: int=None
    ):
        super().__init__()
        self.data_pth = data_pth
        self.col_sele = col_sele
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        df = pl.read_parquet(self.data_pth)
        self.train_df, self.val_df, self.test_df = train_val_test_split(df, self.train_size)

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