import gymnasium as gym
import numpy as np
import numpy.typing as npt
from sty import fg, bg, ef, rs

from collections import Counter
from gym_wordle.utils import to_english, to_array, get_words
from typing import Optional


class WordList(gym.spaces.Discrete):
    """Super class for defining a space of valid words according to a specified
    list.

    The space is a subclass of gym.spaces.Discrete, where each element
    corresponds to an index of a valid word in the word list. The obfuscation
    is necessary for more direct implementation of RL algorithms, which expect
    spaces of less sophisticated form.

    In addition to the default methods of the Discrete space, it implements
    a __getitem__ method for easy index lookup, and an index_of method to
    convert potential words into their corresponding index (if they exist).
    """

    def __init__(self, words: npt.NDArray[np.int64], **kwargs):
        """
        Args:
            words: Collection of words in array form with shape (_, 5), where
              each word is a row of the array. Each array element is an integer
              between 0,...,26 (inclusive).
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        super().__init__(words.shape[0], **kwargs)
        self.words = words

    def __getitem__(self, index: int) -> npt.NDArray[np.int64]:
        """Obtains the (int-encoded) word associated with the given index.

        Args:
            index: Index for the list of words.

        Returns:
            Associated word at the position specified by index.
        """
        return self.words[index]

    def index_of(self, word: npt.NDArray[np.int64]) -> int:
        """Given a word, determine its index in the list (if it exists),
        otherwise returning -1 if no index exists.

        Args:
            word: Word to find in the word list.

        Returns:
            The index of the given word if it exists, otherwise -1.
        """
        try:
            index, = np.nonzero((word == self.words).all(axis=1))
            return index[0]
        except:
            return -1


class SolutionList(WordList):
    """Space for *solution* words to the Wordle environment.

    In the game Wordle, there are two different collections of words:

    * "guesses", which the game accepts as valid words to use to guess the
      answer.
    * "solutions", which the game uses to choose solutions from.

    Of course, the set of solutions is a strict subset of the set of guesses.

    This class represents the set of solution words.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        words = get_words('solution')
        super().__init__(words, **kwargs)


class WordleObsSpace(gym.spaces.Box):
    """Implementation of the state (observation) space in terms of gym
    primitives, in this case, gym.spaces.Box.

    The Wordle observation space can be thought of as a 6x5 array with two
    channels:

      - the character channel, indicating which characters are placed on the
        board (unfilled rows are marked with the empty character, 0)
      - the flag channel, indicating the in-game information associated with
        each character's placement (green highlight, yellow highlight, etc.)

    where there are 6 rows, one for each turn in the game, and 5 columns, since
    the solution will always be a word of length 5.

    For simplicity, and compatibility with stable_baselines algorithms,
    this multichannel is modeled as a 6x10 array, where the two channels are
    horizontally appended (along columns). Thus each row in the observation
    should be interpreted as c0 c1 c2 c3 c4 f0 f1 f2 f3 f4 when the word is
    c0...c4 and its associated flags are f0...f4.
    """

    def __init__(self, **kwargs):
        self.n_rows = 6
        self.n_cols = 5
        self.max_char = 26
        self.max_flag = 4

        low = np.zeros((self.n_rows, 2*self.n_cols))
        high = np.c_[np.full((self.n_rows, self.n_cols), self.max_char),
                     np.full((self.n_rows, self.n_cols), self.max_flag)]

        super().__init__(low, high, dtype=np.int64, **kwargs)


class GuessList(WordList):
    """Space for *guess* words to the Wordle environment.

    This class represents the set of guess words.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        words = get_words('guess')
        super().__init__(words, **kwargs)


class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Character flag codes
    no_char = 0
    right_pos = 1
    wrong_pos = 2
    wrong_char = 3

    def __init__(self):
        super().__init__()

        self.action_space = GuessList()
        self.solution_space = SolutionList()

        self.observation_space = WordleObsSpace()

        self._highlights = {
            self.right_pos: (bg.green, bg.rs),
            self.wrong_pos: (bg.yellow, bg.rs),
            self.wrong_char: ('', ''),
            self.no_char: ('', ''),
        }

        self.n_rounds = 6
        self.n_letters = 5
        self.info = {'correct': False, 'guesses': set()}

    def _highlighter(self, char: str, flag: int) -> str:
        """Terminal renderer functionality. Properly highlights a character
        based on the flag associated with it.

        Args:
            char: Character in question.
            flag: Associated flag, one of:
                - 0: no character (render no background)
                - 1: right position (render green background)
                - 2: wrong position (render yellow background)
                - 3: wrong character (render no background)

        Returns:
            Correct ASCII sequence producing the desired character in the
            correct background.
        """
        front, back = self._highlights[flag]
        return front + char + back

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and returns an initial
        observation.

        Note: The observation space instance should be a Box space.

        Returns:
            state (object): The initial observation of the space.
        """
        self.round = 0
        self.solution = self.solution_space.sample()

        self.state = np.zeros((self.n_rounds, 2 * self.n_letters), dtype=np.int64)

        self.info = {'correct': False, 'guesses': set()}

        return self.state, self.info

    def render(self, mode: str = 'human'):
        """Renders the Wordle environment.

        Currently supported render modes:
        - human: renders the Wordle game to the terminal.

        Args:
            mode: the mode to render with.
        """
        if mode == 'human':
            for row in self.state:
                text = ''.join(map(
                    self._highlighter,
                    to_english(row[:self.n_letters]).upper(),
                    row[self.n_letters:]
                ))
                print(text)
        else:
            super().render(mode=mode)

    def step(self, action):
        """Run one step of the Wordle game. Every game must be previously
        initialized by a call to the `reset` method.

        Args:
            action: Word guessed by the agent.

        Returns:
            state (object): Wordle game state after the guess.
            reward (float): Reward associated with the guess.
            done (bool): Whether the game has ended.
            info (dict): Auxiliary diagnostic information.
        """
        assert self.action_space.contains(action), 'Invalid word!'

        action = self.action_space[action]
        solution = self.solution_space[self.solution]

        self.state[self.round][:self.n_letters] = action

        counter = Counter()
        for i, char in enumerate(action):
            flag_i = i + self.n_letters
            counter[char] += 1

            if char == solution[i]:
                self.state[self.round, flag_i] = self.right_pos
            elif counter[char] <= (char == solution).sum():
                self.state[self.round, flag_i] = self.wrong_pos
            else:
                self.state[self.round, flag_i] = self.wrong_char

        self.round += 1

        correct = (action == solution).all()
        game_over = (self.round == self.n_rounds)

        done = correct or game_over

        reward = 0
        # correct spot
        reward += np.sum(self.state[:, 5:] == 1) * 2

        # correct letter not correct spot
        reward += np.sum(self.state[:, 5:] == 2) * 1

        # incorrect letter
        reward += np.sum(self.state[:, 5:] == 3) * -1

        # guess same word as before
        hashable_action = tuple(action)
        if hashable_action in self.info['guesses']:
            reward += -10
        else:  # guess different word
            reward += 10

        self.info['guesses'].add(hashable_action)

        # for game ending in win or loss
        reward += 10 if correct else -10 if done else 0

        self.info['correct'] = correct

        # observation, reward, terminated, truncated, info
        return self.state, reward, done, False, self.info
