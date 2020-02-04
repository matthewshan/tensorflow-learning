import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

model = keras.models.load_model("ReinforcedLearningTutorial/carpole.h5")

env = gym.make('CartPole-v0')
for i in range(50):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation = observation.reshape([1, 4])
        with tf.GradientTape() as tape:
            output = model(observation).numpy() # Probabilty vector where each element corresponds to an action
            action = np.random.choice(output[0], p=output[0]) #Returns an index where the chances of picking an element corresponds with its value
            temp = output == action #Makes an array of booleans. The choosen action becomes True, and the other elements False
            action = np.argmax(temp) #Grabs the indice of the first occurance of the value True
        observation, reward, done, _ = env.step(action)