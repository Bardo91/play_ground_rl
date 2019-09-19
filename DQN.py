#
# Implementation of TF2 DQN algorithm
#
#
import IPython
import tensorflow as tf
import numpy as np
from collections import deque
from datetime import datetime

class DQN:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]
        self.__iniNet()

    def learn(self,  iterations=100000, 
                    replayBufferSize = 10000, 
                    eGreedy0 = 0.1, 
                    eGreedyFactor=0.99, 
                    batchSize = 32, 
                    learningRate=1e-4, 
                    discountFactor = 0.95,
                    visualize = False):
        
        iter = 0
        replayMemory = deque(maxlen=replayBufferSize)
        
        while iter < iterations:
            ob0 = self.env.reset()
            totalReward = 0
            while True:
                action = 0
                probE = np.random.random()
                if(probE < eGreedy0): # Pick random action
                    action = np.random.random_integers(0,self.n_actions-1)
                else: # Pick best action
                    action, reward = self.predictMaxAction(ob0)
                ob, rew, done, _ = self.env.step(np.asscalar(action))
                if visualize:
                    self.env.render()

                replayMemory.append([ob0, action, rew, ob])
                ob0 = ob

                if len(replayMemory) > 2 * batchSize: # 666 look for how to do it
                    randomIds = np.random.choice(range(len(replayMemory)), size=batchSize, replace=False).tolist()
                    experiences = [replayMemory[i] for i in range(len(randomIds))]
                    self.__trainStep(experiences, done, discountFactor, learningRate)

                totalReward += rew
                if done: # Episode has finished
                    break

            with self.trainWriter:
                tf.summary.scalar("episode_reward", totalReward)

            eGreedy0 *= eGreedyFactor   # Decrease greedy factor

    def predict(self, state):
        state = state.reshape(-1,4)
        r_actions = self.qNet.predict(state)

        return r_actions

    def predictMaxAction(self, state):
        state = state.reshape(-1,4)
        r_actions = self.qNet.predict(state)
        maxAction = np.argmax(r_actions, axis=1)
        maxReward = np.array([r_actions[i,maxAction[i]] for i in range(state.shape[0])])

        return maxAction, maxReward

    def __trainStep(self, experiences, terminal, discountFactor, learningRate):
        y = np.array([[1,0] if exp[2] else [0,1] for exp in experiences], dtype='float64')
        x = np.array([np.array(exp[3]) for exp in experiences])
        if not terminal:
            r_actions = self.predict(x)
            y += np.dot(discountFactor,r_actions) # add max reward
        
        loss = self.qNet.train_on_batch(x, y)
        # with self.trainWriter:
        #     tf.summary.scalar("loss_train", tf.keras.metrics.Mean()(loss))


    def __iniNet(self):
        self.logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.trainWriter = tf.summary.FileWriter(self.logdir)
        self.qNet = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(self.n_observations,)),    ## 666 BATCH SIZE?
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(self.env.action_space.n, activation='sigmoid')
                    ])

        self.optimizer = tf.keras.optimizers.Adam(  learning_rate=0.001,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-07,
                                                    amsgrad=False )


        self.qNet.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])


