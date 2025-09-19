# Importing Libraries 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout
import matplotlib.pyplot as plt


# Loading Dataset

mnist = keras.datasets.mnist
(x_train , y_train) , (x_test , y_test) = mnist.load_data()


# Data Preprocessing 
# Reshaping the data to include a channel dimension (for grayscale)
#(samples,height,width,channels)

x_train = x_train.reshape(60000, 28, 28 ,1)
x_test = x_test.reshape(10000, 28, 28 ,1)

# Normalizing the pixel values ([0,255] -> [0,1])

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# One-hot encoding labels

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("Training data shape: ", x_train.shape)
print("Test data shape: ", x_test.shape)
print("Example one-hot encoded label for y_train: ", y_train[0])
print("Example one-hot encoded label for y_test: ", y_test[0])


#Building CNN Model


model = keras.Sequential([
    # 1st CN Block
    Conv2D(32, kernel_size=(3,3), activation = 'relu'
    , input_shape = (28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    # 2nd CN Block
    Conv2D(64, kernel_size=(3,3), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
    # Flats 2D Features to 1D vector
    Flatten(),
    # Fully connecting layers for classfication
    Dense(128, activation = 'relu'),Dropout(0.45),

    # Output layer (10 Neurons)
    Dense(10, activation = 'softmax')
])

model.summary


# Compiling model

model.compile(optimizer = 'adam' ,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


# Training model

history = model.fit(x_train, y_train,
                    batch_size = 128,
                    epochs = 10,
                    validation_split = 0.1)

# Evaluation 

score = model.evaluate(x_test , y_test , verbose = 0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# Prediction 

image_index = 187
test_image = x_test[image_index]

test_image_batch = np.expand_dims(test_image, axis =0)

prediction = model.predict(test_image_batch)
predicted_digit = np.argmax(prediction)

plt.imshow(test_image.squeeze(),cmap = 'gray')
plt.title(f"Model Prediciton : {predicted_digit}")
plt.axis("off")
plt.show()
