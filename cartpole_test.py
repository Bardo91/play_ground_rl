import gym
from DQN import DQN
import sys
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--iters', type=int, help='an integer for the accumulator', default=1000)
parser.add_argument('--buffer_size', type=int, help='an integer for the accumulator', default=100000)
parser.add_argument('--model_path', type=str, help='an integer for the accumulator')
args = parser.parse_args()

env = gym.make('CartPole-v1')

dqn = DQN(env)

dqn.learn(          iterations=args.iters, 
                    replayBufferSize = args.buffer_size, 
                    eGreedy0 = 0.15, 
                    eGreedyFactor=0.99, 
                    batchSize = 32, 
                    learningRate=1e-3, 
                    discountFactor = 0.95, 
                    visualize=True,
                    saveEachIter = 100,
                    modelPath=args.model_path)

# dqn.load("dqn_300.hdf5")

# Observation space
# [position of cart, velocity of cart, angle of pole, rotation rate of pole

# Action space
# [-1, 1]

env.close()

