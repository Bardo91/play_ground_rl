import gym
from DQN import DQN
import sys
import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--iters', type=int, help='an integer for the accumulator', default=1000)
parser.add_argument('--buffer_size', type=int, help='an integer for the accumulator', default=100000)
parser.add_argument('--model_path', type=str, help='an integer for the accumulator')
parser.add_argument('--train', help='an integer for the accumulator', action='store_true')
parser.add_argument('--non_random', help='an integer for the accumulator', action='store_true')
args = parser.parse_args()

env = gym.make('CartPole-v1')


if args.non_random:
    np.random.seed(1)
    tf.random.set_seed(1)
    env.seed(1)

dqn = DQN(env)

if args.train:
    print("Training")
    dqn.learn(          iterations=args.iters, 
                        replayBufferSize = args.buffer_size, 
                        eGreedy0 = 0.9, 
                        eGreedyFactor=0.99, 
                        batchSize = 32, 
                        learningRate=1e-3, 
                        discountFactor = 0.9, 
                        visualize=True,
                        saveEachIter = 100,
                        modelPath=args.model_path)

else:
    print("Testing")
    dqn.load(args.model_path)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            r_actions = dqn.predict(observation)
            observation, reward, done, info = env.step(np.argmax(r_actions))
            if done:
                break

env.close()

