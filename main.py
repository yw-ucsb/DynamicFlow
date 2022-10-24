import gym
import d4rl

import numpy as np

env = gym.make('halfcheetah-random-v2')

dataset = env.get_dataset()


s = dataset['observations']
a = dataset['actions']
r = dataset['rewards']
end = dataset['terminals']
timeout = dataset['timeouts']



print(timeout)