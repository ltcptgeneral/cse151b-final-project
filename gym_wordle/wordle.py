import gymnasium as gym
import numpy as np
import numpy.typing as npt
from sty import fg, bg, ef, rs

from collections import Counter
from gym_wordle.utils import to_english, to_array, get_words
from typing import Optional
from collections import defaultdict


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
        self.info = {
            'correct': False,
            'guesses': set(),
            'known_positions': np.full(5, -1),  # -1 for unknown, else letter index
            'known_letters': set(),  # Letters known to be in the word
            'not_in_word': set(),  # Letters known not to be in the word
            'tried_positions': defaultdict(set)  # Positions tried for each letter
        }

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
        self.soln_hash = set(self.solution_space[self.solution])

        self.state = np.zeros((self.n_rounds, 2 * self.n_letters), dtype=np.int64)

        self.info = {
            'correct': False,
            'guesses': set(),
            'known_positions': np.full(5, -1),
            'known_letters': set(),
            'not_in_word': set(),
            'tried_positions': defaultdict(set)
        }

        self.simulate_first_guess()

        return self.state, self.info

    def simulate_first_guess(self):
        fixed_first_guess = "rates"
        fixed_first_guess_array = to_array(fixed_first_guess)

        # Simulate the feedback for each letter in the fixed first guess
        feedback = np.zeros(self.n_letters, dtype=int)  # Initialize feedback array
        for i, letter in enumerate(fixed_first_guess_array):
            if letter in self.solution_space[self.solution]:
                if letter == self.solution_space[self.solution][i]:
                    feedback[i] = 1  # Correct position
                else:
                    feedback[i] = 2  # Correct letter, wrong position
            else:
                feedback[i] = 3  # Letter not in word

        # Update the state to reflect the fixed first guess and its feedback
        self.state[0, :self.n_letters] = fixed_first_guess_array
        self.state[0, self.n_letters:] = feedback

        # Update self.info based on the feedback
        for i, flag in enumerate(feedback):
            if flag == self.right_pos:
                # Mark letter as correctly placed
                self.info['known_positions'][i] = fixed_first_guess_array[i]
            elif flag == self.wrong_pos:
                # Note the letter is in the word but in a different position
                self.info['known_letters'].add(fixed_first_guess_array[i])
            elif flag == self.wrong_char:
                # Note the letter is not in the word
                self.info['not_in_word'].add(fixed_first_guess_array[i])

        # Since we're simulating the first guess, increment the round counter
        self.round = 1

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
        assert self.action_space.contains(action), 'Invalid word!'

        guessed_word = self.action_space[action]
        solution_word = self.solution_space[self.solution]

        reward = 0
        correct_guess = np.array_equal(guessed_word, solution_word)

        # Initialize flags for current guess
        current_flags = np.full(self.n_letters, self.wrong_char)

        # Track newly discovered information
        new_info = False

        for i in range(self.n_letters):
            guessed_letter = guessed_word[i]
            if guessed_letter in solution_word:
                # Penalize for reusing a letter found to not be in the word
                if guessed_letter in self.info['not_in_word']:
                    reward -= 2

                # Handle correct letter in the correct position
                if guessed_letter == solution_word[i]:
                    current_flags[i] = self.right_pos
                    if self.info['known_positions'][i] != guessed_letter:
                        reward += 10  # Large reward for new correct placement
                        new_info = True
                        self.info['known_positions'][i] = guessed_letter
                    else:
                        reward += 20  # Large reward for repeating correct placement
                else:
                    current_flags[i] = self.wrong_pos
                    if guessed_letter not in self.info['known_letters'] or i not in self.info['tried_positions'][guessed_letter]:
                        reward += 10  # Reward for guessing a letter in a new position
                        new_info = True
                    else:
                        reward -= 20  # Penalize for not leveraging known information
                    self.info['known_letters'].add(guessed_letter)
                    self.info['tried_positions'][guessed_letter].add(i)
            else:
                # New incorrect letter
                if guessed_letter not in self.info['not_in_word']:
                    reward -= 2  # Penalize for guessing a letter not in the word
                    self.info['not_in_word'].add(guessed_letter)
                    new_info = True
                else:
                    reward -= 15  # Larger penalty for repeating an incorrect letter

        # Update observation state with the current guess and flags
        self.state[self.round, :self.n_letters] = guessed_word
        self.state[self.round, self.n_letters:] = current_flags

        # Check if the game is over
        done = self.round == self.n_rounds - 1 or correct_guess
        self.info['correct'] = correct_guess

        if correct_guess:
            reward += 100  # Major reward for winning
        elif done:
            reward -= 50  # Penalty for losing without using new information effectively
        elif not new_info:
            reward -= 10  # Penalty if no new information was used in this guess

        self.round += 1

        return self.state, reward, done, False, self.info
