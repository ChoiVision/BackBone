import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.dataset import TinyImageNet
from src.util import transforms

class TinyImageNetModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.train= pd.read_csv(self.config.TRAIN_CSV)
        self.val= pd.read_csv(self.config.VALID_CSV)
        self.test= pd.read_csv(self.config.TEST_CSV)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_ds= TinyImageNet(transforms(self.config.IMG_SIZE), self.train['path'], self.train['label'])
            self.valid_ds= TinyImageNet(transforms(self.config.IMG_SIZE), self.val['path'], self.val['label'])

        if stage == 'test':
            self.test_ds= TinyImageNet(transforms, self.test['path'], None)

    def train_dataloader(self):
        return self._dataloader(self.train_ds, True)

    def val_dataloader(self):
        return self._dataloader(self.valid_ds, False)

    def test_dataloader(self):
        return self._dataloader(self.test_ds, False)

    def _dataloader(self, dataset, is_train=False):
        return DataLoader(dataset= dataset,
                          batch_size= self.args.BATCH,
                          pin_memory= True,
                          shuffle= is_train)