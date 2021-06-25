# %%
from posixpath import pardir
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from matplotlib import pyplot
import os
import numpy

from numpy.lib.npyio import load

# %%
# Building dataset with a data builder class

# Flag to control rebuilds
REBUILD_DATA = False

PARENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PARENT_DIR, 'data/archive/SkinDatasets/datasets/train')
WRINKLED_DATA = os.path.join(DATASET_DIR, 'wrinkled')
NON_WRINKLED_DATA = os.path.join(DATASET_DIR, 'no_wrinkles')

print (PARENT_DIR)
print (DATASET_DIR)
print (WRINKLED_DATA)
print (NON_WRINKLED_DATA)
# %%
# Import missing package: numpy
import numpy as np

# %%
# Setting dataset parameters
data_count = {'wrinkled': 0, 'no_wrinkles': 0}
labels = {'wrinkled': 0, 'no_wrinkles': 1}
img_size = (224, 224)
processed_imgs = []
test_percentage = 0.15
training_data = []
testing_data = []

# %%
# Pre processing and building data

for l in labels:
    print ("Processing for ", l, "...")

    for f in os.listdir(os.path.join(DATASET_DIR, l)):
        if 'jpg' in f:

            data_count[l] += 1
            img = load_img (os.path.join(DATASET_DIR, l, f), target_size=img_size)
            processed_imgs.append (img)
    
    print (l, " images found: ", data_count[l], " Shaped to: ", img_to_array(img).shape)
#%%

