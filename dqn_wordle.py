import gym
import gym_wordle
from stable_baselines3 import DQN

env = gym.make("Wordle-v0")
done = False

print(env)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=100)
model.save("dqn_wordle")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_wordle")

state = env.reset()

while not done:

    action, _states = model.predict(state, deterministic=True)

    state, reward, done, info = env.step(action)