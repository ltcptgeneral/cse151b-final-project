import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import re
import copy


class LetterGuessingEnv(gym.Env):
    """
    Custom Gymnasium environment for a letter guessing game with a focus on forming
    valid prefixes and words from a list of valid Wordle words. The environment tracks
    the current guess prefix and validates it against known valid words, ending the game
    early with a negative reward for invalid prefixes.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, valid_words, seed=None):
        self.action_space = spaces.Discrete(26)
        self.observation_space = spaces.Box(low=0, high=1, shape=(26*2 + 26*4,), dtype=np.int32)

        self.valid_words = valid_words  # List of valid Wordle words
        self.target_word = ''  # Target word for the current episode
        self.valid_words_str = ' '.join(self.valid_words) + ' '
        self.letter_flags = None
        self.letter_positions = None
        self.guessed_letters = set()
        self.guess_prefix = ""  # Tracks the current guess prefix

        self.reset()

    def clone_state(self):
        # Clone the current state
        return {
            'target_word': self.target_word,
            'letter_flags': copy.deepcopy(self.letter_flags),
            'letter_positions': copy.deepcopy(self.letter_positions),
            'guessed_letters': copy.deepcopy(self.guessed_letters),
            'guess_prefix': self.guess_prefix,
            'round': self.round
        }

    def set_state(self, state):
        # Restore the state
        self.target_word = state['target_word']
        self.letter_flags = copy.deepcopy(state['letter_flags'])
        self.letter_positions = copy.deepcopy(state['letter_positions'])
        self.guessed_letters = copy.deepcopy(state['guessed_letters'])
        self.guess_prefix = state['guess_prefix']
        self.round = state['round']

    def step(self, action):
        letter_index = action  # Assuming action is the letter index directly
        position = len(self.guess_prefix)  # The next position in the prefix is determined by its current length
        letter = chr(ord('a') + letter_index)

        reward = 0
        done = False

        # Check if the letter has already been used in the guess prefix
        if letter in self.guessed_letters:
            reward = -1  # Penalize for repeating letters in the prefix
        else:
            # Add the new letter to the prefix and update guessed letters set
            self.guess_prefix += letter
            self.guessed_letters.add(letter)

            # Update letter flags based on whether the letter is in the target word
            if self.target_word[position] == letter:
                self.letter_flags[letter_index, :] = [1, 0]  # Update flag for correct guess
            elif letter in self.target_word:
                self.letter_flags[letter_index, :] = [0, 1]  # Update flag for correct guess wrong position
            else:
                self.letter_flags[letter_index, :] = [0, 0]  # Update flag for incorrect guess

            reward = 1  # Reward for adding new information by trying a new letter

            # Update the letter_positions matrix to reflect the new guess
            if position == 4:
                self.letter_positions[:, :] = 1
            else:
                self.letter_positions[:, position] = 0
                self.letter_positions[letter_index, position] = 1

        # Use regex to check if the current prefix can lead to a valid word
        if not re.search(r'\b' + self.guess_prefix, self.valid_words_str):
            reward = -5  # Penalize for forming an invalid prefix
            done = True  # End the episode if the prefix is invalid

        # guessed a full word so we reset our guess prefix to guess next round
        if len(self.guess_prefix) == len(self.target_word):
            self.guess_prefix = ''
            self.round += 1

        # end after 3 rounds of total guesses
        if self.round == 3:
            # reward = 5
            done = True

        obs = self.get_obs()

        if reward < -5:
            print(obs, reward, done)
            exit(0)

        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.target_word = random.choice(self.valid_words)
        # self.target_word_encoded = self.encode_word(self.target_word)
        self.letter_flags = np.ones((26, 2), dtype=np.int32)
        self.letter_positions = np.ones((26, 4), dtype=np.int32)
        self.guessed_letters = set()
        self.guess_prefix = ""  # Reset the guess prefix for the new episode
        self.round = 0
        return self.get_obs(), {}

    def encode_word(self, word):
        encoded = np.zeros((26,))
        for char in word:
            index = ord(char) - ord('a')
            encoded[index] = 1
        return encoded

    def get_obs(self):
        return np.concatenate([self.letter_flags.flatten(), self.letter_positions.flatten()])

    def render(self, mode='human'):
        pass  # Optional: Implement rendering logic if needed
