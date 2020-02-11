import gym
import numpy as np

env = gym.make("CartPole-v0")
obs = env.reset()
env.render()

image = env.render(mode="rgb_array")
print(image.shape)

action = 1
obs, reward, done, info = env.step(action)
print("Observation: " + str(obs))
print("Reward: " + str(reward))
print("Done: " + str(done))
print("Info: " + str(info))


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


total = []
for episode in range(10):
    episode_reward = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            break
    total.append(episode_reward)

print("Mean: " + str(np.mean(total)) + "  Standard Deviation: " + str(np.std(total)) + "  Min: " + str(np.min(total)) + "  Max: " + str(np.max(total)))

