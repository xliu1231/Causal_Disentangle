
import os

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd



class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_path, attr_path, attr=[], transform=None):
    
        df = pd.read_csv(attr_path, sep="\s+", skiprows=1, index_col=0)
        df = df.replace(-1, 0)
        self.data_path = data_path
        self.attr_path = attr_path
        self.img_names = df.index.values
        self.target = df[attr].values if attr else df.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.target[index]
        return img, label

    def __len__(self):
        return self.target.shape[0]