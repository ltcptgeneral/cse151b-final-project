import gym
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import wordle_gym
import numpy as np
from tqdm import tqdm

def train (model, env, total_timesteps = 100000): 
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("dqn_wordle")

def test(model, env, test_num=1000):

    total_correct = 0

    for i in tqdm(range(test_num)):

        model = DQN.load("dqn_wordle")

        env = gym.make("wordle-v0")
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            
    return total_correct / test_num

if __name__ == "__main__":
    
    env = gym.make("wordle-v0")
    model = DQN("MlpPolicy", env, verbose=0)
    print(env)
    print(model)

    train(model, env, total_timesteps=500000)
    print(test(model, env))