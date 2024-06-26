#!/usr/bin/env python
# coding: utf-8

# ### Choose GPU settings, import libraries

# In[1]:



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
num_GPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) 

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model

import sys

sys.setrecursionlimit(100000000)
print(sys.getrecursionlimit())

config = tf.compat.v1.ConfigProto()

#config = tf.ConfigProto() 
config.gpu_options.allow_growth=True 
session = tf.compat.v1.Session(config=config)
tensorflow.compat.v1.keras.backend.set_session(session)


# In[2]:


from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm_notebook as tqdm
# Config the matplotlib backend as plotting inline in IPython
#get_ipython().run_line_magic('matplotlib', 'inline')

import keras_helper
from keras_helper import *
from skimage.transform import resize


# ### Some training parameters

# In[3]:


h,w=64,64
nepochs=75
wt_path = 'wts4' #Where to store network weights
batch_size = 32

if (not os.path.isdir(wt_path)):
    os.mkdir(wt_path)


# ### Read experimental diffraction data and reconstructed images

# In[4]:


data_diffr = np.load('data/20191008_39_diff.npz')['arr_0']


# In[5]:


print(data_diffr.shape)
#plt.matshow(np.log10(data_diffr[0,0]))

data_diffr_red = np.zeros((data_diffr.shape[0],data_diffr.shape[1],64,64), float)
for i in tqdm(range(data_diffr.shape[0])):
    for j in range(data_diffr.shape[1]):
        data_diffr_red[i,j] = resize(data_diffr[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
        data_diffr_red[i,j] = np.where(data_diffr_red[i,j]<3,0,data_diffr_red[i,j])

image = data_diffr[0,0,32:-32, 32:-32]
print(image.shape)

# In[6]:
#amp = np.load('../expt_data/s26_data/20191008_30_10nm_amp.npz')['arr_0']
#ph = np.load('../expt_data/s26_data/20191008_39_10nm.npz')['arr_0']
real_space = np.load('data/20191008_39_amp_pha_10nm_full.npy')
amp = np.abs(real_space)
ph = np.angle(real_space)
