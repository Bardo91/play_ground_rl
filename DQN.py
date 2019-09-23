#
# Implementation of TF2 DQN algorithm
#
#
import IPython
import tensorflow as tf
import numpy as np
from collections import deque
from datetime import datetime

from ReplayBuffer import ReplayBuffer

from memory_profiler import profile
import os

class DQN:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]
        self.isNetInit = False
    
    def learn(self,
                    iterations=1000, 
                    replayBufferSize = 1000, 
                    eGreedy0 = 1.0, 
                    eGreedyFactor=0.995, 
                    batchSize = 32, 
                    learningRate=1e-3, 
                    discountFactor = 0.9,
                    targetNet = False,
                    visualize = False,
                    saveEachIter = 100, 
                    modelPath = None):
                
        self.targetNet = targetNet
        self.__iniNet(learningRate)
        if modelPath:
            self.qNet.load_weights(modelPath)
            if self.targetNet:
                self.tNet.load_weights(modelPath)


        iter = 0
        replayMemory = ReplayBuffer(size=replayBufferSize)
        
        train_label = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.mkdir("checkpoint_"+train_label)
        writer = tf.summary.create_file_writer("./train_dqn/train_"+train_label)

        while iter < iterations:
            ob0 = self.env.reset()
            totalReward = 0
            losses = np.array([])
            while True:
                action = 0
                probE = np.random.random()

                if(probE < eGreedy0): # Pick random action
                    action = np.random.random_integers(0,self.n_actions-1)
                else: # Pick best action
                    action, _ = self.predictMaxAction(ob0)
                
                ob, rew, done, _ = self.env.step(np.asscalar(action))
                if visualize:
                    self.env.render()

                replayMemory.add(ob0, action, rew, ob, done)
                ob0 = ob

                if replayMemory.can_sample(batchSize): 
                    experiences = replayMemory.sample(batchSize)
                    loss = self.__trainStep(experiences, done, discountFactor, learningRate)
                    np.append(losses, tf.reduce_mean(loss))

                totalReward += rew
                if done: # Episode has finished
                    break

            with writer.as_default():
                tf.summary.scalar("episode_reward", totalReward, step=iter)
                tf.summary.scalar("epsilon_greedy", eGreedy0, step=iter)
                tf.summary.scalar("train_loss", tf.reduce_mean(losses), step=iter)
                writer.flush()

            if iter % saveEachIter == 0:
                self.qNet.save_weights("checkpoint_"+train_label+"/dqn_"+str(iter)+".hdf5")

            eGreedy0 *= eGreedyFactor   # Decrease greedy factor
            if eGreedy0 < 0.02:
                eGreedy0 = 0.02
            
            iter +=1
            
            if self.targetNet:
                self.tNet.set_weights(self.qNet.get_weights()) 

    def load(self, modelPath):
        self.qNet = tf.keras.models.load_model(modelPath)

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
        o_x = experiences[0] # np.array([np.array(exp[0]) for exp in experiences])               # Prev state
        o_a = experiences[1] #np.array([exp[1] for exp in experiences], dtype='float64')        # Actions
        o_r = experiences[2] #np.array([exp[2] for exp in experiences], dtype='float64')        # Rewards
        o_xf = experiences[3] #np.array([exp[3] for exp in experiences], dtype='float64')       # Next state
        o_done  = experiences[4] #np.array([exp[4] for exp in experiences], dtype='float64')    # Terminal step

        if self.targetNet:
            q_update = o_r + discountFactor * np.max(self.tNet.predict(o_xf), axis=-1)*(1-o_done) # add max reward
        else:
            q_update = o_r + discountFactor * np.max(self.qNet.predict(o_xf), axis=-1)*(1-o_done) # add max reward
        # Generate a target value that is the same output for non-trainable actions and different for trainable ones
        target_value = self.predict(o_x)

        for i in range(len(o_a)):
            target_value[i][int(o_a[i])] = q_update[i]

        # loss = self.qNet.train_on_batch(o_x, target_value)

        loss_value, grads = self.grad(self.qNet, o_x, target_value)
        self.optimizer.apply_gradients(zip(grads, self.qNet.trainable_variables))

        return loss_value

        # with self.trainWriter:
        #     tf.summary.scalar("loss_train", tf.keras.metrics.Mean()(loss))

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = tf.keras.losses.MSE(model(inputs), targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    def __iniNet(self, learningRate):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0: 
            tf.config.experimental.set_memory_growth(physical_devices[0], True)


        self.isNetInit = True
        self.qNet = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(self.n_observations,)),    ## 666 BATCH SIZE?
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(self.env.action_space.n, activation='linear')
                    ])


        if self.targetNet:
            self.tNet = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(self.n_observations,)),    ## 666 BATCH SIZE?
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(self.env.action_space.n, activation='linear')
                    ])

        self.optimizer = tf.keras.optimizers.Adam(  learning_rate=learningRate,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-07,
                                                    amsgrad=False )


        # self.qNet.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

