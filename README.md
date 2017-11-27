# Using-VGG-to-detect-defrauds-in-High-Reflection-Curved-Surface
Using the VGG Convloutional Neural Network to detect the defrauds in  High-Reflection-Curved Surface.
It is hard to tell the highlight and defrauds in a High-Reflection-Curved Surface.
So we try to use the VGG to solve this problem.
We use 試験片画像作成.py to read the images and then, according to their folder,we add a label for data.
We use 試験片学習.py to build the model and do the DeepLearning, I use Keras with Tensorflow Backends here.
We use visualize.py to visualize the images(after convolution etc)
Retrain. py is used for testing model and adjusting code.
ROF.py is used to pre-processing.
