# Gym-Wordle

An OpenAI gym compatible environment for training agents to play Wordle.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/8514041/152437216-d78e85f6-8049-4cb9-ae61-3c015a8a0e4f.gif"><br/>
  <em>User-input demo of the environment</em>
</p>

## Installation

My goal is for a minimalist package that lets you install quickly and get on
with your research. Installation is just a simple call to `pip`:

```
$ pip install gym_wordle
```

### Requirements

In keeping with my desire to have a minimalist package, there are only three
major requirements:

* `numpy`
* `gym`
* `sty`, a lovely little package for stylizing text in terminals

## Usage

The basic flow for training agents with the `Wordle-v0` environment is the same
as with gym environments generally:

```Python
import gym
import gym_wordle

eng = gym.make("Wordle-v0")

done = False
while not done:
    action = ...  # RL magic
    state, reward, done, info = env.step(action)
```

If you're like millions of other people, you're a Wordle-obsessive in your own
right. I have good news for you! The `Wordle-v0` environment currently has an
implemented `render` method, which allows you to see a human-friendly version
of the game. And it isn't so hard to set up the environment to play for
yourself. Here's an example script:

```Python
from gym_wordle.utils import play

play()
```

## Documentation

Coming soon!

## Examples

Coming soon!

## Citing

If you decide to use this project in your work, please consider a citation!

```bibtex
@misc{gym_wordle,
  author = {Kraemer, David},
  title = {An Environment for Reinforcement Learning with Wordle},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidNKraemer/Gym-Wordle}},
}
```
