import numpy as np
from PIL import Image
import random as rnd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, AveragePooling2D, Flatten
from sklearn.model_selection import train_test_split
from keras import Model
from keras.optimizers.schedules import ExponentialDecay

from model import resnet_like_model, resnet_trainable_conv
from utils import get_filenames, get_xy_pairs
from sequence import MushroomsSequence

# This module was downloaded from github.com/bckenstler/CLR. Make sure 
# to add the path to this module's folder in your PC to the PYTHONPATH
# environment variable.
from clr_callback import CyclicLR

# Instantiate the model with the proper number of output units.
model = resnet_trainable_conv(output_units=9)

# Instantiate the optimizer by giving it an exponential decay learning
# rate schedule.
# init_lr = 0.001
# decay_steps = 500
# decay_rate = 0.6
# schedule = ExponentialDecay(init_lr, decay_steps, decay_rate, staircase=True)
optimizer = keras.optimizers.Adam()

# Compile the model with the above optimizer and a loss function for
# multi-class classification.
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
    metrics='accuracy')

# Get all the filenames in the Mushrooms folder and then make 95%/5%
# train/validation split, making sure the distribution of the different
# labels remain roughly the same among the two sets (this is done by
# specifying the stratify parameter in the train_test_split function).
folder = './Mushrooms'
all_files = get_filenames()
x, y = get_xy_pairs(all_files)
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=.05, stratify=y)

# Define the Sequence objects so as to load batches of input images and 
# labels in chuncks (both for train and validation).
tr_files = [f'{y}_{x}' for (y, x) in zip(y_tr, x_tr)]
seq_tr = MushroomsSequence(folder, files=tr_files)
val_files = [f'{y}_{x}' for (y, x) in zip(y_val, x_val)]
seq_val = MushroomsSequence(folder, files=val_files)

# It is recommended that the step size is set from 2 to 8 times the
# number of iterations in a single epoch, i.e. the number of batches
# (in our case it's 200).
clr = CyclicLR(step_size=400., max_lr=.015, mode='exp_range',
    gamma=0.99994)
chk = ModelCheckpoint('checkpoint.h5', monitor='val_accuracy',
    save_best_only=True, save_weights_only=True)

# Add some callbacks.
callbacks = [
    chk,
    clr
]

history = model.fit(x=seq_tr, epochs=100, callbacks=callbacks,
    validation_data=seq_val)

# import matplotlib.pyplot as plt
# plt.plot(callbacks[1].history['lr'], callbacks[1].history['loss'])
# plt.show()

#model.save_weights('best_weights.h5')