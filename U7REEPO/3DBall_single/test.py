env_name = "Unity Environment.exe"   # Name of the Unity environment binary to launch

import matplotlib.pyplot as plt
import numpy as np
import sys

from gym_unity.envs import UnityEnv

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env = UnityEnv(env_name, worker_id=1)

# Examine environment parameters
print("OBS Shape: " + str(env.observation_space.shape))
print("Action Shape: " + str(env.action_space.shape))
print(str(env.action_space.sample()))

# Reset the environment
initial_observation = env.reset()

if len(env.observation_space.shape) == 1:
    # Examine the initial vector observation
    print("Agent state looks like: \n{}".format(initial_observation))
else:
    # Examine the initial visual observation
    print("Agent observations look like:")
    if env.observation_space.shape[2] == 3:
        plt.imshow(initial_observation[:,:,:])
    else:
        plt.imshow(initial_observation[:,:,0])
    plt.show()
print(env.action_space.shape[0])
for episode in range(10):
    initial_observation = env.reset()
    done = False
    episode_rewards = 0

    while not done:
        a = env.action_space.sample()
        observation, reward, done, info = env.step(a)
        episode_rewards += reward
        print("Action: "+ str(a))
    print("Total reward this episode: {}".format(episode_rewards))