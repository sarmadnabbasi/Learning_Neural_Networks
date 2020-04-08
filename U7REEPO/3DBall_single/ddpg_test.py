import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from gym_unity.envs import UnityEnv

#env = gym.make('MountainCarContinuous-v0')
env = UnityEnv("Unity Environment.exe", 0, flatten_branched=True)
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="TBLogsTest")
model.learn(total_timesteps=0)


del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_mountain")

obs = env.reset()
rewards = 0
episode_reward = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_reward = episode_reward + rewards
    print("Reward: " + str(rewards))
    if(dones):
        print("Episode Reward: " + str(episode_reward))
        episode_reward = 0
    env.render()
