import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TinyImageNet(Dataset):
    def __init__(self, transform, path, label=None):
        super().__init__()
        self.transform= transform
        self.path= path
        self.label= label

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        image= cv2.imread(self.path[index])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= self.transform(image= image)['image']

        if self.label is not None:
            label= self.label[index]

            return image, label
        
        return image


        
