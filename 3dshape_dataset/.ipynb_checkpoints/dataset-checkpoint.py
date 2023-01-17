from torch.utils.data import Dataset
import torch
import numpy as np
import random
import h5py
from PIL import Image
import csv


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


def sample_3dshape_dataset(factors_distribution, data_size, data_path, label_path):
    """
    factors_distribution: A dictionary containing factors and distribution of their values.
                          For factors not in this dictionary, we will sample them randomly.
    data_size: Total size of the dataset.
    save_name: save the selected indices with the given name
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
    with open(label_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for idx in indices:
            image = data['images'][idx]
            label = data['labels'][idx]
            img = Image.fromarray(image)
            img.save(data_path +'/'+str(idx)+'.jpg')
            writer.writerow(list(label))
    return






# class ShapesDataset(Dataset):
#     """
#     data_path: Path of the sampled indices
#     concept_interest: A list of indices indicating what factors to be included in the labels.
#                       The indices follow _FACTORS_IN_ORDER.
#                       If it is None, we will include all the labels.
#     transform: It should at least contains torchvision.transforms.ToTensor()
#     """

#     def __init__(self, data_path, concept_interest=None, transform=None):
#         self.indices = np.load(data_path)
#         self.transform = transform
#         # TODO: using h5 might be time consuming, considering save all images to jpeg
#         data = h5py.File(_DATA_PATH, 'r')
#         self.images = data['images']
#         self.labels = data['labels']
#         if concept_interest is None:
#             self.concept_interest = [0, 1, 2, 3, 4, 5]
#         else:
#             self.concept_interest = concept_interest

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, index):
#         idx = self.indices[index]
#         image = self.images[idx]
#         attrs = self.labels[idx]

#         if self.transform is not None:
#             image = self.transform(image)

#         labels = []
#         for factor in self.concept_interest:
#             if factor not in [0, 1, 2, 3, 4, 5]:
#                 raise Exception(f'Unknown factor index {factor}')
#             # TODO: return index of the value instead of the value
#             labels.append(attrs[factor])
#         labels = torch.Tensor(labels)

#         return image, labels


