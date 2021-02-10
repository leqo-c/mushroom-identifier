# mushroom-identifier
Transfer Learning experiments on Kaggle's "Mushrooms classification - Common genus's images" data set using ResNet and EfficientNet.

The data set can be found [here](https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images) (download the data as it is and double check that all images are inside a folder called "Mushrooms").

## Most relevant files
- `fitv2.py`: Start training the model (either ResNet or EfficientNet) on the training data set. At first, the fully-connected layer we add on top of the pretrained network is trained for a small number of epoches. Then, in the fine-tuning phase, the training process is resumed and applied also to the pretrained convnet's topmost convolutional layer;
- `model.py`: File containing functions that return ready-to-use instances of ResNet and EfficientNet; 
- `evalmodel.py`: Evaluate the trained model on a handful of images (a checkpoint is given as an example).
