import gym
import gym_tictactoe
env = gym.make('TicTacToe-v1')
env.render(mode=None)
env.init(symbols=[-1, 1]) # Define users symbols

user = 0
done = False
reward = 0

# Reset the env before playing
state = env.reset()