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

from model import resnet_like_model, efficient_net
from utils import get_filenames, get_xy_pairs
from sequence import MushroomsSequence

# Instantiate the model with the proper number of output units.
model = efficient_net(output_units=9)

# Define the optimizer.
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
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=.1, stratify=y)

# Define the Sequence objects so as to load batches of input images and 
# labels in chuncks (both for train and validation).
tr_files = [f'{y}_{x}' for (y, x) in zip(y_tr, x_tr)]
seq_tr = MushroomsSequence(folder, files=tr_files)
val_files = [f'{y}_{x}' for (y, x) in zip(y_val, x_val)]
seq_val = MushroomsSequence(folder, files=val_files)

# We train the classifier we just put on top of resnet for a sufficient
# number of epochs. It is fundamental that we carry out this step before
# applying any fine-tuning on resnet's topmost convolutional layer.
history = model.fit(x=seq_tr, epochs=7, validation_data=seq_val,
    verbose=1)


# --- FINE-TUNING PHASE ---

# Make the topmost convolutional layer trainable. This allows us to 
# tweak the high-level convolutional features and adapt them to our 
# Mushroom dataset.
eff_net = model.layers[1]
top_conv = eff_net.get_layer('top_conv')
top_conv.trainable = True

# Set all the batch normalization layers as untrainable. From Tensorflow
# 2.0, this will also force them to run in inference mode.
for l in eff_net.layers:
    if 'bn' in l.name:
        l.trainable = False

# We need to recompile the model to make sure the changes we have just
# made actually take place.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy', metrics='accuracy')

# Add some callbacks.
chk = ModelCheckpoint('checkpoint.h5', monitor='val_accuracy',
    save_best_only=True, save_weights_only=True)

callbacks = [
    chk
]

# Resume training from the point we previuosly stopped at.
history_ft = model.fit(x=seq_tr, epochs=100, validation_data=seq_val,
    initial_epoch=history.epoch[-1], verbose=2)

#model.save_weights('best_weights.h5')