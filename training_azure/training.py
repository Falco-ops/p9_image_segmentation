print('print importing lib...')

import argparse
from azureml.core import Run
from azureml.core import Dataset
import joblib
import os
import numpy as np
import pandas as pd
import random
#import cv2

import tensorflow
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.python.keras.optimizers import Adadelta, Nadam
from tensorflow.python.keras.models import Model, load_model
#from tensorflow.python.keras.utils import multi_gpu_model, plot_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping#, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.losses import binary_crossentropy

from tensorflow.python.keras.callbacks import Callback

from sklearn.model_selection import train_test_split

from dilatednet import DilatedNet
from multiclassunet import Unet

print('lib imported...')


#get scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument('--augmented-data', type=str, dest='augmented_data')
args = parser.parse_args()

save_folder = args.augmented_data

run = Run.get_context()

#set parameters
batch_size = 2
samples = 50000
steps = samples//batch_size
img_height, img_width = 256, 256
classes = 8
filters_n = 64

#load data
print('data...')
image_path= os.path.join(save_folder, 'image')
mask_path= os.path.join(save_folder, 'mask')

#divison of labelling
cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

#mounted point to directory
image_dir = image_path
mask_dir = mask_path


#extract list of pictures 
print('making picture list...')
image_list = []
for root, dirs, files in os.walk(image_dir):
    for name in files:
        image_list.append(os.path.join(root, name))

#extract mask with label
mask_list = []
for root, dirs, files in os.walk(mask_dir):
    for name in files:
        mask_list.append(os.path.join(root, name))
        
print("split..")

def train_test_val_split(X, Y, split=(0.2, 0.1), shuffle=True):
    """Split dataset into train/val/test subsets by 70:20:10(default).
    
    Args:
      X: List of data.
      Y: List of labels corresponding to data.
      split: Tuple of split ratio in `test:val` order.
      shuffle: Bool of shuffle or not.
      
    Returns:
      Three dataset in `train:test:val` order.
    """
    
    assert len(X) == len(Y), 'The length of X and Y must be consistent.'
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, 
        test_size=(split[0]+split[1]), shuffle=shuffle)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, 
        test_size=split[1], shuffle=False)
    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val) 

print("class gen...")

class seg_gen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        #make a array of length batch size containing random position of photo
        idx = np.random.randint(0, 10, batch_size)
        batch_x, batch_y = [], []
        drawn = 0
        
        for i in idx:
            #import original image /255 to normalize and avoid big number for compute power
            _image = image.img_to_array(image.load_img(f'{image_list[i]}', target_size=(img_height, img_width)))/255.
            #import mask
            img = image.img_to_array(image.load_img(f'{mask_list[i]}', grayscale=True, target_size=(img_height, img_width)))
            #assess number of different label
            labels = np.unique(img)
            #ignore picture withtout features
            if len(labels) < 3:
                idx = np.random.randint(0, 50000, batch_size-drawn)
                continue
            img = np.squeeze(img)
            #create a tensor of dimension image with one plan for each category
            mask = np.zeros((img.shape[0], img.shape[1], 8))
            for i in range(-1, 34):
                if i in cats['void']:
                    mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
                elif i in cats['flat']:
                    mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
                elif i in cats['construction']:
                    mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
                elif i in cats['object']:
                    mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
                elif i in cats['nature']:
                    mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
                elif i in cats['sky']:
                    mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
                elif i in cats['human']:
                    mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
                elif i in cats['vehicle']:
                    mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
            mask = np.resize(mask,(img_height*img_width, 8))
            batch_y.append(mask)
            batch_x.append(_image)
            drawn += 1
        return np.array(batch_x), np.array(batch_y)
    
print('coef and loss...')

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss\

print('model...')

unet = DilatedNet(256, 256, 8,use_ctx_module=True, bn=True)
unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='models/pdilated.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
callbacks = [tb, mc, es]

print('making input...')
train_image, train_mask, val_image, val_mask = train_test_split(image_list, mask_list, test_size=0.15) 

train_gen = seg_gen(train_image[:10], train_mask[:10], batch_size)
valid_gen = seg_gen(val_image[:10], val_mask[:10], batch_size)

print('fit...')
unet.fit_generator(train_gen, epochs=5, callbacks=callbacks, validation_dat=valid_gen)

print('Saving final weights')
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'dilated.h5')
unet.save_weights(model_file)

run.complete()
