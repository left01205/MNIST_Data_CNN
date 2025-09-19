
-----

# MNIST Handwritten Digit Recognition with CNNs 🧠

This repository provides two distinct implementations of a Convolutional Neural Network (CNN) designed to classify handwritten digits from the famous MNIST dataset. The primary goal of this project is to serve as an educational tool, offering a side-by-side comparison of a high-level framework implementation with a fundamental, from-scratch version.

Whether you want to see a practical application or understand what happens "under the hood" of a deep learning library, this repository has you covered.

## Key Features

  * **Dual Implementations**: See the same problem solved with two different approaches.
  * **Practical Application**: A standard, efficient model built with **TensorFlow/Keras**.
  * **Educational Deep Dive**: A from-scratch model built using only **NumPy** to demystify the core mechanics of a CNN, including forward and backward propagation.
  * **Detailed Code**: Both versions are commented to explain key concepts, data preprocessing steps, and model architecture choices.
  * **Easy to Run**: Simple scripts to train and evaluate both models.

-----

## Implementations

### 1\. TensorFlow/Keras Version (`MNIST_CNN_Keras.py`)

This version represents the standard, practical approach to building a CNN. It leverages the power and simplicity of the Keras API within TensorFlow to build an efficient and highly accurate classifier.

  * **Technology**: Python, TensorFlow, Keras
  * **Performance**: Achieves **\>99% accuracy** on the test set after training.
  * **Best for**: Understanding the modern workflow for building and deploying deep learning models.

### 2\. NumPy (From-Scratch) Version (`MNIST_CNN_Primitive.py`)

This implementation is purely for educational purposes. It builds the entire CNN from the ground up using only NumPy for numerical operations. Every layer (`Conv2D`, `MaxPooling2D`, `Dense`) and the entire backpropagation algorithm is written from scratch.

  * **Technology**: Python, NumPy
  * **Performance**: Significantly slower than the TensorFlow version. It demonstrates a working implementation but is not optimized for speed.
  * **Best for**: Gaining a deep, fundamental understanding of how a CNN works mathematically.

-----

## Project Structure

```
.
├── MNIST_CNN_Keras.py             # The practical implementation using TensorFlow/Keras
├── MNIST_CNN_Primitive.py         # The educational from-scratch implementation using NumPy
├── requirements.txt               # Python dependencies for the project
└── README.md                      # You are here!
```

-----

## Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

  * Python 3.8 or higher
  * pip package manager

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/left01205/MNIST_Data_CNN_git
    cd mnist-cnn-repo
    ```

2.  **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

-----

## Usage

You can run each implementation directly from your terminal.

### Running the TensorFlow/Keras Model

This script will load the data via Keras, build the model, train it for 10 epochs, and evaluate its accuracy.

```sh
python MNIST_CNN_Keras.py
```

### Running the NumPy (From-Scratch) Model

This script will first download the original MNIST dataset files. It then builds and trains the from-scratch model. **Note: This will be much slower than the TensorFlow version.**

```sh
python MNIST_CNN_Primitive.py
```

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
