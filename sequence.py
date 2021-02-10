import math
import numpy as np
import pandas as pd
from os import listdir
from keras.utils import Sequence, to_categorical

from utils import get_filenames, read_images_from_files, get_xy_pairs

class MushroomsSequence(Sequence):

    def __init__(self, images_dir, files=None, batch_size=32):
        """Returns an instance of the MushroomsSequence class for the
        images stored inside the images_dir folder. The list of the
        exact files to be read can be optionally specified through the
        `files` parameter.

        Args:
            images_dir (string): Path to the images' folder.
            files (list, optional): The explicit list of files to be
                read. If files = None, all files inside the images_dir
                folder will be read.
                Defaults to None.
            batch_size (int, optional): The number of images to be
                returned after a call to the `__getitem__` method.
                Defaults to 32.
        """
        self.classes = sorted(listdir(images_dir))
        self.filenames = files if files != None else get_filenames()
        # We shuffle the filenames so that batches will end up having
        # different mushroom species inside them.
        np.random.shuffle(self.filenames)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        from_ = idx * self.batch_size
        to_ = (idx + 1) * self.batch_size

        batch_x, batch_y = get_xy_pairs(self.filenames[from_:to_])

        x_images = read_images_from_files(batch_x)
        indexes = [self.classes.index(b) for b in batch_y]
        y_volumes = to_categorical(indexes, num_classes=len(self.classes))

        return x_images, y_volumes