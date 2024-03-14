from gym.envs.registration import register
from .wordle import WordleEnv

register(
    id='Wordle-v0',
    entry_point='gym_wordle.wordle:WordleEnv'
)
