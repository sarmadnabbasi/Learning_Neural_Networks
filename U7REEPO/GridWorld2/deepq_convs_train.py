from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from gym_unity.envs import UnityEnv

def main():
    logger.configure('./tensorBoard_Pong_GW')
    #logger.make_output_format()
    #logger.TensorBoardOutputFormat()
    env = UnityEnv("Unity Environment.exe", 0, use_visual=True, uint8_visual=True, flatten_branched=True)
    env = bench.Monitor(env, logger.get_dir())
    #env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(6e5),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        checkpoint_freq=1000,
        checkpoint_path='./tensorBoard_Pong_GW',  # Change to save model in a different directory
        gamma=0.99,
        #load_path="GW1_model.pkl"

    )
    print("Saving model to unity_model.pkl")
    #model.save('GW1_model.pkl')
    env.close()

if __name__ == '__main__':
    main()