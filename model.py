import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Model
from tensorflow.keras import layers
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, Dense, Flatten, Dropout

def efficient_net(output_units=20):
    """Return a modified instance of the EfficientNetB7 model where the
    last fully-connected layer has been replaced with a Dense layer with
    a number of units equal to 'output_units'.

    Args:
        output_units (int, optional): The number of output units in the
            final fully-connected layer. Defaults to 20.

    Returns:
        keras.Model: An instance of the model.
    """

    # Setting include_top=False results in the deletion of the model's 
    # last three layers (GlobalAveragePooling2D, Dropout and Dense).
    eff_net = keras.applications.EfficientNetB7(
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # We want to freeze resnet's weights before attaching new layers.
    # This way we will leverage the features learned on the imagenet
    # dataset and we will only need to tweak the final fully-connected
    # layer to our specific needs.
    eff_net.trainable = False

    inputs = Input(shape=(224, 224, 3))
    
    # By setting training=False, we are asking efficientnet to run in
    # inference mode. This will be important for future fine-tuning.
    x = eff_net(inputs, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)
    x = Dense(output_units, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def resnet_like_model(output_units=20):
    """Return a modified instance of the ResNet50 model where the last
    fully-connected layer has been replaced with a Dense layer with a
    number of units equal to 'output_units'.

    Args:
        output_units (int, optional): The number of output units in the
            final fully-connected layer. Defaults to 20.

    Returns:
        keras.Model: An instance of the model.
    """

    # Setting include_top=False results in the deletion of the model's 
    # last two layers (7x7 avg_pool + Dense 1000).
    resnet = keras.applications.ResNet50(
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # We want to freeze resnet's weights before attaching new layers.
    # This way we will leverage the features learned on the imagenet
    # dataset and we will only need to tweak the final fully-connected
    # layer to our specific needs.
    resnet.trainable = False

    inputs = Input(shape=(224, 224, 3))

    # By setting training=False, we are asking resnet to run in
    # inference mode. This will be important for future fine-tuning.
    x = resnet(inputs, training=False)

    x = AveragePooling2D(pool_size=7)(x)
    x = Flatten()(x)
    x = Dense(output_units, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def resnet_trainable_conv(output_units=9):
    """Return a modified instance of the ResNet50 model where the last
    fully-connected layer has been replaced with a Dense layer with a
    number of units equal to 'output_units' and the last convolutional
    layer is trainable.

    Args:
        output_units (int, optional): The number of output units in the
            final fully-connected layer. Defaults to 9.

    Returns:
        keras.Model: An instance of the model.
    """

    # Setting include_top=False results in the deletion of the model's 
    # last two layers (7x7 avg_pool + Dense 1000).
    resnet = keras.applications.ResNet50(
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3))
    )

    # We want to freeze resnet's weights before attaching new layers.
    # This way we will leverage the features learned on the imagenet
    # dataset and we will only need to tweak the final fully-connected
    # layer and the final convolutional layer.
    resnet.trainable = False
    
    last_conv = resnet.get_layer('conv5_block3_3_conv')
    last_conv.trainable = True
    batch_norm = resnet.get_layer('conv5_block3_3_bn')
    batch_norm.trainable = True

    x = resnet.output
    x = AveragePooling2D(pool_size=7)(x)
    x = Flatten()(x)
    x = Dense(9, activation='softmax')(x)

    model = Model(inputs=resnet.input, outputs=x)
    return model