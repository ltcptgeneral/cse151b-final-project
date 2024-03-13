# %%
from stable_baselines3 import DQN
import numpy as np
import wordle.state
import gym

# %%
env = gym.make("WordleEnvFull-v0")

print(env)

# %%
total_timesteps = 100000
model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# %%
def test(model):

    end_rewards = []

    for i in range(1000):
        
        state = env.reset()

        done = False

        while not done:

            action, _states = model.predict(state, deterministic=True)

            state, reward, done, info = env.step(action)
            
        end_rewards.append(reward == 0)
        
    return np.sum(end_rewards) / len(end_rewards)

# %%
model.save("dqn_wordle")

# %%
model = DQN.load("dqn_wordle")

# %%
print(test(model))


