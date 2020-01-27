import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

#Based on the following article: 
#https://medium.com/@hamza.emra/reinforcement-learning-with-tensorflow-2-0-cca33fead626

def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r) #NP: Return an array of zeros with the same shape and type as a given array.
    running_add = 0
    for t in reversed(range(0, r.size)): #The reversed() function returns an iterator that accesses the given sequence in the reverse order.
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#Creates layers of the model
model = keras.Sequential([
    keras.layers.Dense(32, input_dim = 4, activation='relu'),
    keras.layers.Dense(2, activation= "softmax")
])
model.build()

#Specify optimizer
optimizer = keras.optimizers.Adam(learning_rate = 0.01)

#Loss Function
compute_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


#Gradients of backpropagation
gradBuffer = model.trainable_variables
for i, grad in enumerate(gradBuffer):
    gradBuffer[i] = grad * 0

env = gym.make('CartPole-v0')
episodes = 2000
scores = []
update_every = 5

for e in range(episodes):
    # reset the enviroment
    s = env.reset()
    
    ep_memory = []
    ep_score = 0
    done = False
    while not done:
        env.render()
        s = s.reshape([1, 4])
        with tf.GradientTape() as tape:
            #Forward pass
            logits = model(s)
            a_dist = logits.numpy()
            #Choose random action with p = action dist
            action = np.random.choice(a_dist[0], p=a_dist[0])
            action = np.argmax(a_dist == action)
            loss = compute_loss([action], logits)
        # Make the choosen action
        s, reward, done, _ = env.step(action)
        ep_score += reward
        if done:
            reward -= 10
        grads = tape.gradient(loss, model.trainable_variables)
        ep_memory.append([grads, reward])
    scores.append(ep_score)
    #Discount the rewards
    ep_memory = np.array(ep_memory)
    ep_memory[:,1] = discount_rewards(ep_memory[:,1])

    for grad, reward in ep_memory:
        for i, grad in enumerate(grads):
            gradBuffer[i] += grad * reward

    if e % update_every == 0:
        optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
        for i, grad in enumerate(gradBuffer):
            gradBuffer[i] = grad * 0
    
    if e % 100 == 0:
        print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))