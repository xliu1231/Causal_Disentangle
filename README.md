# Causal_Disentangle

## How to generate synthetic 3D shape dataset

### Introduction 

There are 6 attributes in total, each has several possible values.

'floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'

Here we use the object_hue and shape as factors of interests.

There are mainly three files related to the generative process.


> Data Sampler

-  Causal_Disentangle/3dshape_dataset/dataset_3d_shape.py  
- /Causal_Disentangle/prepare_3d_shape.ipynb
- /Causal_Disentangle/datasets.py


Use the jupyternotebook as a reference. 

We first create a folder to store the generated images. Then we use a csv file to store the attributes and labels. 
labels are structured as ["label_name", actual_label(integer)]

