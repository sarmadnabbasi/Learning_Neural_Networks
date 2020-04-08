import gym
from baselines import deepq
from gym_unity.envs import UnityEnv

import time

def main():
    env = UnityEnv("Unity Environment.exe", 0, use_visual=True, uint8_visual=True, flatten_branched=True)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        total_timesteps=0,
        load_path ="GW1_model.pkl"
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            #env.render()
            obs, rew, done, _ = env.step(model(obs[None])[0])
            episode_rew += rew

        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()