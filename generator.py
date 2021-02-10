from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from PIL import ImageFile


def get_generator():
    """Return an ImageDataGenerator object with some default parameters
        that are related to data augmentation, preprocessing and
        validation split.

    Returns:
        `keras.preprocessing.image.ImageDataGenerator`: An image data
            generator object, which can be subsequently used to process
            input data through methods such as `flow_from_directory`.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    gen = ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=preprocess_input,
        validation_split=.1
    )

    return gen