import gym
from DQN import DQN

env = gym.make('CartPole-v0')

dqn = DQN(env)

dqn.learn(iterations=100000, visualize=True)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = dqn.predict(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
        
env.close()
