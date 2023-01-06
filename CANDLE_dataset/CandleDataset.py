from PIL import Image
# import torchvision.transforms as T
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

"""
Labels:
'cube','sphere''red','blue',
'indoor','playground','outdoor','bridge','city square','hall','grassland',
'garage','street','beach','station','tunnel','moonlit grass','dusk city', 'skywalk','garden',
'left light','middle light','right light'

"""
class entangled_Candle(Dataset):
    """Load entangled dataset for training"""
    def __init__(self, data_path, attr_path, attr = [], ratio, transform=None):
        df = pd.read_csv(attr_path, sep=" ", index_col=0)
        df = df.replace(-1, 0)
        self.data_path = data_path
        self.attr_path = attr_path
        self.entangled_names, self.disentangled_names,self.img_names = self.entangled(attr_path,ratio)
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
        return len(self.img_names)
    
    def entangled(self,attr_path,ratio):
        df = pd.read_csv(attr_path, sep=" ", index_col=0)
        df = df.replace(-1, 0)
        all_img_names = df.index.values
        target = df.values

        entangled_img = []
        all_disentangled_img = []
        disentangled_img = []
        for i, name in enumerate(all_img_names):
            if target[i,0] == target[i,2]:
                entangled_img.append(name)
            else:
                all_disentangled_img.append(name)

        dis_num = round(len(entangled_img)/ratio) - len(entangled_img)
        disentangled_img = random.sample(all_disentangled_img, dis_num)
        all_imgs = entangled_img + disentangled_img

        return entangled_img, disentangled_img, all_imgs

    
class CandleDataset(Dataset):
    """Custom Dataset for loading CANDLE generated images"""

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

train_dataset = CandleDataset(data_path="/cmlscratch/margot98/Causal_Disentangle/CANDLE_dataset/images",
                              attr_path='CANDLE_label_test.txt',
                              attr=['cube','sphere'],
                              transform=transform)

train_dataset = entangled_Candle(data_path="/cmlscratch/margot98/Causal_Disentangle/CANDLE_dataset/images",
                              attr_path='CANDLE_label_test.txt',
                            ratio = 0.7,
                              attr=['cube','sphere'],
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=12,
                          shuffle=True,
                          num_workers=4) 