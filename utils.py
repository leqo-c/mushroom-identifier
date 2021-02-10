import os
import numpy as np
from keras.applications import resnet50
from PIL import Image, ImageFile

def get_filenames(folder='./Mushrooms'):
    """Get a list of the filenames of all the images inside the given
    folder. Their format is "{label}_{original_filename}" where label is
    the image's class name and original_filename is the full path to the
    file (starting from the given folder).

    Args:
        folder (string, optional): The folder where the images are
        stored. The folder must contain a subfolder for each possible
        label and subfolders should be named accordingly. Defaults to
        './Mushrooms'.

    Returns:
        [array-like]: The list of the filenames as described above.
    """
    result = []
    labels = sorted(os.listdir(folder))
    for l in labels:
        subfolder = f'{folder}/{l}'
        result += [
            l
            + '_'
            + os.path.join(subfolder, elem) for elem in os.listdir(subfolder)
        ]
    
    return result

def read_images_from_files(filenames, target_shape=(224, 224)):
    """Reads images from the given list of paths and resizes them to the
    given shape. It also applies the required preprocessing for resNet50
    model.

    Args:
        filenames (array-like): The list of filepaths to read images
            from.
        target_shape (tuple, optional): A tuple indicating the target
            shape images need to be resized to. Defaults to (224, 224).

    Returns:
        [array-like]: A numpy array of size (m, target_shape, 3), where
            images are represented as numpy arrays themselves and are
            stacked vertically along the first axis.
            m is the number of images (the length of filenames).
    """
    image_list = []
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    for filename in filenames:
        im = Image.open(filename).resize(target_shape)
        # Handle the special case of grayscale images.
        if im.mode != 'RGB':
            im = im.convert("RGB")
        image_list.append(resnet50.preprocess_input(np.asarray(im)))
        
    return np.stack(image_list, axis=0)

def get_xy_pairs(filenames):
    """Given a list of filenames having the format
    "{label}_{original_filename}", separate the {label} parts from the
    {original_filename} parts by putting them in two separate lists
    (while still maintaining their original order).

    Args:
        filenames (array-like): List of strings containing the filenames
            whose parts we wish to separate.

    Returns:
        tuple: A pair containing the two lists. The first one contains
            the {original_filename} parts, while the second one contains
            the {label} parts.
    """
    bi_split = lambda s, sep: s.split(sep=sep, maxsplit=1)
    batch_x = [bi_split(f, '_')[1] for f in filenames]
    batch_y = [bi_split(f, '_')[0] for f in filenames]
    return batch_x, batch_y