from PIL import Image
import torchvision.transforms as T
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

"""
Labels:
'cube','sphere','cylinder','cone','torus','red','blue','yellow','purple','orange','size1.5','size2','size2.5',
'rotation0','rotation15','rotation30','rotation45','rotation60','rotation90','indoor','playground','outdoor',
'bridge','city square','hall','grassland','garage','street','beach','station','tunnel','moonlit grass','dusk city',
'skywalk','garden','left light','middle light','right light'

"""
class CandleDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_path, attr_path, attr=[], transform=None):
    
        df = pd.read_csv(attr_path, sep=" ", index_col=0)
        df = df.replace(-1, 0)
        self.data_path = data_path
        self.attr_path = attr_path
        self.img_names = df.index.values
        self.target = df[attr].values if attr else df.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path,str(self.img_names[index])+'.png'))
        
        if self.transform is not None:
            img = self.transform(img)
            img = img[:3, :, :]
        
        label = self.target[index]
        return img, label

    def __len__(self):
        return self.target.shape[0]

image_size = 64
transform = T.Compose([
    T.Resize([image_size, image_size]),
    T.ToTensor()
])

train_dataset = CandleDataset(data_path="/cmlscratch/margot98/Causal_dir/CANDLE/",
                              attr_path='CANDLE_label.txt',
                              attr=['cube','sphere'],
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=12,
                          shuffle=True,
                          num_workers=4) 