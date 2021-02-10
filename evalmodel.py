import os
import numpy as np
from model import resnet_like_model
from utils import read_images_from_files

model = resnet_like_model(output_units=9)
model.load_weights('20201223.h5')

labels = np.array([
    'agaricus',
    'amanita',
    'boletus',
    'cortinarius',
    'entoloma',
    'hygrocybe',
    'lactarius',
    'russula',
    'suillus'
])

test_files = sorted([f'test-pics/{e}' for e in os.listdir('test-pics')])
x = read_images_from_files(test_files)
preds = model.predict(x)

# Print the top-n predictions, along with their confidence scores.
n = 3
for i in range(len(preds)):
    indices = np.argsort(-preds[i])
    top_n = list(zip(labels[indices[:n]], preds[i][indices[:n]]))
    print(f'{test_files[i]}: -> {top_n}')