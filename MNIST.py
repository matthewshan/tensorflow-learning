import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

# Creates our neural network model and its layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Input layer
    keras.layers.Dense(128, activation="relu"), # rectified linear
    keras.layers.Dense(10, activation="softmax") #softmax - Probability of something - like a probability vector
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train our model
model.fit(train_images, train_labels, epochs=5) #Epochs: How of then the model well see the image

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Acc: ", test_acc)

#Gets the last layer of the neurons
prediction = model.predict(test_images) #Input is a list of lists that represents the input layer
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: "  + class_names[np.argmax(prediction[i])])
    plt.show()