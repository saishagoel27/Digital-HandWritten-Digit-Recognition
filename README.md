Project 1 - Digital Handwritten Digit Recognition
1. Introduction:
This project is a handwritten digit recognition system using the MNIST dataset. The goal is to recognize and classify handwritten digits (0-9) from the famous MNIST dataset using machine learning techniques, specifically with neural networks. The model is trained to predict the correct digit from input images of handwritten digits.
2. Dataset Used:
Dataset Name: MNIST (Modified National Institute of Standards and Technology)
Description: This dataset consists of 70,000 labeled images of handwritten digits. It includes 60,000 training images and 10,000 test images, each of size 28x28 pixels, in grayscale.
3. Model Overview:
The model is built using TensorFlow and Keras, a deep learning framework.
It consists of:
 i. Input Layer: Images of size 28x28 pixels.
 ii. Hidden Layers: A combination of convolutional layers and fully connected layers.
 iii. Output Layer: 10 neurons (one for each digit, 0-9) with a softmax activation function to predict the class probability.
4. Technologies Used:
Languages: Python
Libraries:
TensorFlow, Keras, NumPy, Matplotlib, Pandas, Jupyter Notebooks
