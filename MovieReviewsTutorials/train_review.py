import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

#Get the word index for the data given 
word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#Creates a reverse dictionary (Number: String) of word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#Decodes inputs into human readable format
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#How to train your data
def train():
    (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) # most frequent 10000 words
    # Preprocesses the data so that to have all training data the same length at 250
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)
    #ANN Model
    #Create the layers
    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000, 16)) #Word vectors for each word in 16D Space. Groups word vectors based on potential relationships
    model.add(keras.layers.GlobalAveragePooling1D()) #Takes the average if each dimension and puts it in to a list (16 in this case)
    model.add(keras.layers.Dense(16, activation="relu")) #Dense layer that accepts the the list of 16
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    # Compiles our model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #Loss function will calculate how off the output was from 0 or 1

    # Validate the model
    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)

    #Save the model we just trained
    model.save("model.h5")

#train()

def encode_review(word_list):
    encoded = [1]
    for word in word_list:
        encoded.append(word_index.get(word.lower(), 2))
    return encoded

model = keras.models.load_model("model.h5")

with open("fmab.txt", encoding="utf-8") as myfile:
    review = []
    for line in myfile.readlines():
        review.extend(line.replace(",", "").replace("'", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", "").strip().split(" "))
encode = encode_review(review)
encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
prediction = model.predict(encode)
print(" ".join(review))
print(encode)
print(prediction[0])

"""
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " +str(test_labels[0]))
print(results)
"""

