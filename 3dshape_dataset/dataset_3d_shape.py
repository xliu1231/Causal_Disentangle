from torch.utils.data import Dataset
import torch
import numpy as np
import random
import h5py
from PIL import Image
import csv
import os


_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4,
                          'orientation': 15}
_VALUES_PER_FACTOR = {'floor_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'wall_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'object_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'scale': [0.75, 0.82142857, 0.89285714, 0.96428571, 1.03571429,
                                1.10714286, 1.17857143, 1.25],
                      'shape': [0., 1., 2., 3.],
                      'orientation': [-30., -25.71428571, -21.42857143, -17.14285714,
                                      -12.85714286, -8.57142857, -4.28571429, 0.,
                                      4.28571429, 8.57142857, 12.85714286, 17.14285714,
                                      21.42857143, 25.71428571, 30.]
                      }
_DATA_PATH = '/cmlscratch/bangan/project/dataset/3dshape/3dshapes.h5'


def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
      factors: np array shape [6,batch_size].
               factors[i]=factors[i,:] takes integer values in
               range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
      indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


def sample_3dshape_dataset(factors_distribution, data_size, data_path, label_path, new_label=None):
    """
    factors_distribution: A dictionary containing factors and distribution of their values.
                          For factors not in this dictionary, we will sample them randomly.
    data_size: Total size of the dataset.
    data_path: A directory to save the generated images
    label_path: A csv file that saves label information
    new_label:["new_label_name", new_label_identifier (interger)]
    """
    factors = np.zeros([len(_FACTORS_IN_ORDER), data_size], dtype=np.int32)
    for factor, name in enumerate(_FACTORS_IN_ORDER):
        if name in factors_distribution:
            num_choices = _NUM_VALUES_PER_FACTOR[name]
            factors[factor] = random.choices(np.arange(0, num_choices),
                                             weights=factors_distribution[name], k=data_size)
        else:
            num_choices = _NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, data_size)
    
    indices = get_index(factors)
    #np.save(save_name, indices)
    
    # edited by XL: save sampled data to jpeg to save time
    indices.sort()
    data = h5py.File(_DATA_PATH, 'r')
    
    if os.path.exists(label_path):
        mode = 'a'
    else:
        mode = 'w'
    
    with open(label_path, mode, newline='') as file:
        writer = csv.writer(file)
        header = ['img_name', 'floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        if new_label:
            header.append(new_label[0])
        writer.writerow(header)
        for idx in indices:
            image = data['images'][idx]
            label = list(data['labels'][idx])
            img = Image.fromarray(image)
            img.save(data_path +'/'+str(idx)+'.jpg')
            
            # add identifier for each data
            label.insert(0, str(idx)+'.jpg')
            label.append(new_label[1])
            
            writer.writerow(label)
    return









