# 🧠 MNIST Handwritten Digit Recognition
*A simple neural network that recognizes handwritten digits*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-93.6%25-brightgreen.svg)

*Learning machine learning fundamentals with the classic MNIST dataset*

</div>

## 🎯 What This Project Does

This is a beginner-friendly implementation of handwritten digit recognition using:
- **MNIST Dataset**: 70,000 images of handwritten digits (0-9)
- **Simple Neural Network**: Single dense layer with 10 neurons
- **TensorFlow/Keras**: For building and training the model
- **Basic Prediction**: Classify individual digit images

Perfect for understanding the fundamentals of neural networks and image classification!

## 🚀 Quick Demo

```python
# Load a test image
plt.matshow(x_test[0])  # Shows a handwritten digit

# Make prediction
prediction = model.predict(x_test_flattened[0:1])
predicted_digit = np.argmax(prediction[0])

print(f"Predicted Digit: {predicted_digit}")
# Output: Predicted Digit: 7
```

## 📁 Project Structure

```
├── Untitled.ipynb          # Main notebook with all code
├── my_model.weights.h5     # Saved model weights
└── README.md              # You're reading it!
```

## 🧬 Model Architecture

**Simple but effective:**
```
Input: 784 pixels (28×28 flattened)
    ↓
Dense Layer: 10 neurons (sigmoid activation)
    ↓
Output: 10 probabilities (one per digit)
```

## 📊 What's Actually Implemented

✅ **Data Loading**: MNIST dataset from TensorFlow  
✅ **Data Preprocessing**: Reshape (28×28 → 784) and normalize (0-255 → 0-1)  
✅ **Model Creation**: Single dense layer with sigmoid activation  
✅ **Training**: 20 epochs with Adam optimizer  
✅ **Model Persistence**: Save/load weights to file  
✅ **Single Prediction**: Classify one test image  
✅ **Basic Visualization**: Display images with matplotlib  

## 🛠️ How to Run

### Prerequisites
```bash
pip install tensorflow numpy matplotlib jupyter
```

### Steps
1. **Clone/Download** this repository
2. **Open** `Untitled.ipynb` in Jupyter Notebook
3. **Run all cells** from top to bottom
4. **Watch** the model train for 20 epochs
5. **See** prediction on test image

### Training Output (What You'll See)
```
Epoch 1/20  - accuracy: 0.8145 - loss: 0.7249
Epoch 10/20 - accuracy: 0.9306 - loss: 0.2515  
Epoch 20/20 - accuracy: 0.9358 - loss: 0.2325
```

**Final Result: ~93.6% accuracy** 🎯

## 📚 Code Walkthrough

### 1. Load Data
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

### 2. Preprocess Images
```python
# Flatten 28x28 images to 784-length vectors
x_train_final = x_train.reshape(len(x_train), 784)
# Normalize pixel values to 0-1 range
x_train_final = x_train_final / 255
```

### 3. Build Model
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(784,)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 4. Train
```python
model.fit(x_train_final, y_train, epochs=20)
```

### 5. Save Model
```python
model.save_weights('./my_model.weights.h5')
```

### 6. Make Predictions
```python
predictions = model.predict(x_test_final)
predicted_digit = np.argmax(predictions[0])
```

## 🎓 What You'll Learn

- **Neural Network Basics**: How dense layers work
- **Image Preprocessing**: Flattening and normalization
- **Training Process**: Watching loss decrease and accuracy improve
- **Model Persistence**: Saving and loading trained weights
- **Making Predictions**: Converting model output to classifications

## 🚀 Next Steps (Ideas for Enhancement)

- [ ] Add more layers to improve accuracy
- [ ] Implement CNN (Convolutional Neural Network)
- [ ] Visualize training progress with plots
- [ ] Test on your own handwritten digits
- [ ] Add confusion matrix analysis
- [ ] Experiment with different optimizers
- [ ] Try data augmentation techniques

## ❓ Common Questions

**Q: Why only 93.6% accuracy?**  
A: This is a deliberately simple model! CNNs can achieve 99%+ accuracy.

**Q: Can I test my own handwriting?**  
A: Not directly implemented, but you could preprocess your own 28×28 images.

**Q: Why sigmoid activation?**  
A: Simple choice for learning. ReLU or softmax might work better.

## 📖 Learning Resources

- [MNIST Database Info](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Beginner Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Neural Networks Explained](https://www.youtube.com/watch?v=aircAruvnKk)

## 🤝 Contributing

This is a learning project! Feel free to:
- Improve the model architecture
- Add visualizations
- Create better documentation
- Test different approaches

## 📄 License

Free to use for learning and experimentation!

---

<div align="center">

*A simple first step into the world of machine learning* 🚀

</div>
