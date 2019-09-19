#
# Implementation of TF2 DQN algorithm
#
#

import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, env):
        self.__iniNet()
        self.env = env
        self.n_actions = self.env.action_space.n

    def learn(self,  iterations=100000, 
                    replayBufferSize = 10000, 
                    eGreedy0 = 1.0, 
                    eGreedyFactor=0.99, 
                    batchSize = 32, 
                    learningRate=1e-4, 
                    discountFactor = 0.95,
                    visualize = False):
        
        iter = 0
        replayMemory = []
        
        while iter < iterations:
            ob0 = self.env.reset()
            while True:
                action = 0
                probE = np.random.random()
                if(probE < eGreedy0): # Pick random action
                    action = np.random.random_integers(self.n_actions)
                else: # Pick best action
                    action = bestAction

                ob, rew, done, _ = self.env.step(action)
                if visualize:
                    self.env.render()

                replayMemory.append([ob0, action, rew, ob])
                ob0 = ob

                if len(replayMemory) > 10 * batchSize: # 666 look for how to do it
                    randomIds = np.random.choice(range(len(replayMemory)), size=batchSize, replace=False)
                    experiences = np.array(replayMemory[randomIds])
                    self.__trainStep(experiences, done)

                if done: # Episode has finished
                    break

            eGreedy0 *= eGreedyFactor   # Decrease greedy factor

    def predict(self, state):
        maxReward = 0
        maxAction = 0
        for i in range(self.n_actions):
            rew = self.qNet.predict(np.array[state, action])
            if rew > maxReward:
                maxReward = rew
                maxAction = i
        
        return i

    def __trainStep(self, experiences, terminal, discountFactor, learningRate):
        y = experiences[:,2]
        x = experiences[:,0:2]
        if not terminal:
            y += discountFactor*0 # sum max reward  666 Do not understand
        
        qNet.train_on_batch(x, y)

    def __maxReward(self):
        pass

    def __iniNet(self):
        self.qNet = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(self.env.observation_space.n)),    ## 666 BATCH SIZE?
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(self.env.action_space.n, activation='sigmoid')
                    ])

        self.optimizer = tf.keras.optimziers.Adam(  learning_rate=0.001,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-07,
                                                    amsgrad=False )


        self.qNet.compile(optimizer=self.optimizer, loss='MSE')


