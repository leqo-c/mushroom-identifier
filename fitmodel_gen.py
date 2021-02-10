import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from clr_callback import CyclicLR
from generator import get_generator
from model import resnet_like_model


# Instantiate the model with the proper number of output units.
model = resnet_like_model(output_units=9)

optimizer = keras.optimizers.Adam()

# Compile the model with the above optimizer and a loss function for
# multi-class classification.
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
    metrics='accuracy')

# Set the name of the directory where images should be extracted from.
directory = 'Mushrooms'

# Get a default instance of the image data generator, then produce two
# separated data streams (one for training, one for validation).
gen = get_generator()
train_data = gen.flow_from_directory(directory, target_size=(224, 224),
    color_mode='rgb', batch_size=32, shuffle=True, subset='training')
val_data = gen.flow_from_directory(directory, target_size=(224, 224),
    color_mode='rgb', batch_size=32, shuffle=True, subset='validation')

# It is recommended that the step size is set from 2 to 8 times the
# number of iterations in a single epoch, i.e. the number of batches
# (in our case it's 200).
clr = CyclicLR(step_size=400., max_lr=.015, mode='exp_range',
    gamma=0.99994)
chk = ModelCheckpoint('model_aug.h5', monitor='val_accuracy',
    save_best_only=True, save_weights_only=True)

# Add some callbacks.
callbacks = [
    chk,
    clr
]

# Train the model.
model.fit(train_data, epochs=100, callbacks=callbacks, validation_data=val_data,
    verbose=2)