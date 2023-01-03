from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range

import argparse
import os
import numpy as np
import PIL
import scipy.io as sio
from six.moves import range
from sklearn.utils import extmath
import json
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class candleDataset():

    def __init__(self,data_path,num_img,transform=None):
        self.object_code = {'cube': 0, 'sphere': 1}
        self.color_code = {'red': 0, 'blue': 1}
        self.scene_code = {'indoor':0, 'playground':1, 'outdoor':2,'bridge':3,'city square':4, 'hall':5,'grassland':6,'garage':7,'street':8,'beach':9,'station':10,'tunnel':11,
          'moonlit grass':12, 'dusk city':13, 'skywalk':14,'garden':15}
        self.light_code = {'left':0, 'middle':1,'right':2}
        self.num_img = num_img
        self.path = data_path
        

    def get_latent(self,obj):
        temp = [0 for _ in range(4)]
        temp[0] = self.object_code[obj['object']]
        temp[1] = self.color_code[obj['color']]
        temp[2] = self.scene_code[obj['scene']]
        temp[3] = self.light_code[obj['light']]
        return temp


    def get_label(self,latents):
        num_obj = 2
        num_color = 2
        num_scene = 16
        num_light = 3
        sum_num = num_obj + num_color + num_scene + num_light
        temp = [0 for _ in range(sum_num)]
        temp[latents[0]] = 1
        temp[num_obj + latents[1]] = 1
        temp[num_obj+num_color + latents[2]] = 1
        temp[num_obj + num_color + num_scene + latents[3]] = 1
        return temp
    
    def get_imgs(self):
        labels = []
        img_name = []

        for _ in range(self.num_img):
            if not os.path.isfile(self.path + str(_) + '.json'):
                continue
            with open(self.path + str(_) + '.json') as fp:
                img_name.append(_)
                obj = json.load(fp)
                ob = {}
                ob['scene'] = obj['scene']
                te = obj['objects'][list(obj['objects'].keys())[0]]
                ob['object'] = te['object_type']
                ob['color'] = te['color']
                ob['light'] = obj['lights']
                labels.append(self.get_label(self.get_latent(ob)))
        return img_name,np.array(labels)

def parse():
    parser = argparse.ArgumentParser(description =
        "label generator for candle dataset")
    parser.add_argument('--outfile', default='./CANDLE_label_test.txt', type=str,help = 'file to store the labels')
    parser.add_argument("--datapath", default= "/cmlscratch/margot98/Causal_Disentangle/candle-simulator/images/", type=str,help="The path to the folder stroing the data.")
    parser.add_argument('--num-img', default=4, type=int)
    return parser.parse_args(args=[])

def main():
    global args
    args = parse()
    
    Candle = candleDataset(args.datapath,args.num_img)
    image_name,labels_np = Candle.get_imgs()
    lables_pd = pd.DataFrame({
            'Index Title':image_name,
            'cube' : labels_np[:,0],
             'sphere':labels_np[:,1],
             'red':labels_np[:,2],
             'blue':labels_np[:,3],
             'indoor':labels_np[:,4],
             'playground':labels_np[:,5],
             'outdoor':labels_np[:,6],
             'bridge':labels_np[:,7],
             'city square':labels_np[:,8],
             'hall':labels_np[:,9],
             'grassland':labels_np[:,10],
             'garage':labels_np[:,11],
             'street':labels_np[:,12],
             'beach':labels_np[:,13],
             'station':labels_np[:,14],
             'tunnel':labels_np[:,15],
             'moonlit grass':labels_np[:,16],
             'dusk city':labels_np[:,17],
             'skywalk':labels_np[:,18],
             'garden':labels_np[:,19],
             'left light':labels_np[:,20],
             'middle light':labels_np[:,21],
             'right light':labels_np[:,22]})

    lables_pd.index = lables_pd["Index Title"]
    
    del lables_pd["Index Title"]
    
    lables_pd.to_csv('CANDLE_label_test.txt',sep=' ')
    
if __name__ == "__main__":
    main()

