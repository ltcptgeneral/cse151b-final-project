import re
import string

import numpy as np

from stable_baselines3 import PPO, DQN
from letter_guess import LetterGuessingEnv
import torch

def load_valid_words(file_path='wordle_words.txt'):
    """
    Load valid five-letter words from a specified text file.

    Parameters:
    - file_path (str): The path to the text file containing valid words.

    Returns:
    - list[str]: A list of valid words loaded from the file.
    """
    with open(file_path, 'r') as file:
        valid_words = [line.strip() for line in file if len(line.strip()) == 5]
    return valid_words


class AI:
    def __init__(self, vocab_file, model_file, num_letters=5, num_guesses=6, use_q_model=False):
        self.vocab_file = vocab_file
        self.num_letters = num_letters
        self.num_guesses = 6

        self.vocab, self.vocab_scores, self.letter_scores = self.get_vocab(self.vocab_file)
        self.best_words = sorted(list(self.vocab_scores.items()), key=lambda tup: tup[1])[::-1]

        self.domains = None
        self.possible_letters = None

        self.use_q_model = use_q_model
        if use_q_model:
            # we initialize the same q env as the model train ONLY to simplify storing/calculating the gym state, not used to control the game at all
            self.q_env = LetterGuessingEnv(load_valid_words(vocab_file))
            self.q_env_state, _ = self.q_env.reset()

            # load model
            self.q_model = PPO.load(model_file)

        self.reset("")

    def solve_eval(self, results_callback):
        num_guesses = 0
        while [len(e) for e in self.domains] != [1 for _ in range(self.num_letters)]:
            num_guesses += 1
            if self.use_q_model:
                self.freeze_state = self.q_env.clone_state()

            # sample a word, this would use the q_env_state if the q_model is used
            word = self.sample(num_guesses)

            # get emulated results
            results = results_callback(word)
            if self.use_q_model:
                self.q_env.set_state(self.freeze_state)
                # step the q_env to match the guess we just made
                for i in range(len(word)):
                    char = word[i]
                    action = ord(char) - ord('a')
                    self.q_env_state, _, _, _, _ = self.q_env.step(action)

            self.arc_consistency(word, results)
        return num_guesses, word

    def solve(self):
        num_guesses = 0
        while [len(e) for e in self.domains] != [1 for _ in range(self.num_letters)]:
            num_guesses += 1
            if self.use_q_model:
                self.freeze_state = self.q_env.clone_state()

            # sample a word, this would use the q_env_state if the q_model is used
            word = self.sample(num_guesses)

            print('-----------------------------------------------')
            print(f'Guess #{num_guesses}/{self.num_guesses}: {word}')
            print('-----------------------------------------------')

            print(f'Performing arc consistency check on {word}...')
            print(f'Specify 0 for completely nonexistent letter at the specified index, 1 for existent letter but incorrect index, and 2 for correct letter at correct index.')
            results = []

            # Collect results
            for l in word:
                while True:
                    result = input(f'{l}: ')
                    if result not in ['0', '1', '2']:
                        print('Incorrect option. Try again.')
                        continue
                    results.append(result)
                    break

            if self.use_q_model:
                self.q_env.set_state(self.freeze_state)
                # step the q_env to match the guess we just made
                for i in range(len(word)):
                    char = word[i]
                    action = ord(char) - ord('a')
                    self.q_env_state, _, _, _, _ = self.q_env.step(action)

            self.arc_consistency(word, results)
        return num_guesses, word

    def arc_consistency(self, word, results):
        self.possible_letters += [word[i] for i in range(len(word)) if results[i] == '1']

        for i in range(len(word)):
            if results[i] == '0':
                if word[i] in self.possible_letters:
                    if word[i] in self.domains[i]:
                        self.domains[i].remove(word[i])
                else:
                    for j in range(len(self.domains)):
                        if word[i] in self.domains[j] and len(self.domains[j]) > 1:
                            self.domains[j].remove(word[i])
            if results[i] == '1':
                if word[i] in self.domains[i]:
                    self.domains[i].remove(word[i])
            if results[i] == '2':
                self.domains[i] = [word[i]]

    def reset(self, target_word):
        self.domains = [list(string.ascii_lowercase) for _ in range(self.num_letters)]
        self.possible_letters = []

        if self.use_q_model:
            self.q_env_state, _ = self.q_env.reset()
            self.q_env.target_word = target_word

    def sample(self, num_guesses):
        """
        Samples a best word given the current domains
        :return:
        """
        # Compile a regex of possible words with the current domain
        regex_string = ''
        for domain in self.domains:
            regex_string += ''.join(['[', ''.join(domain), ']', '{1}'])
        pattern = re.compile(regex_string)

        # From the words with the highest scores, only return the best word that match the regex pattern
        max_qval = float('-inf')
        best_word = None
        for word, _ in self.best_words:
            # reset the state back to before we guessed a word
            if pattern.match(word) and False not in [e in word for e in self.possible_letters]:
                if self.use_q_model and num_guesses == 3:
                    self.q_env.set_state(self.freeze_state)
                    # Use policy to grade word
                    # get the state and action pairs
                    curr_qval = 0

                    for l in word:
                        action = ord(l) - ord('a')
                        q_val, _, _ = self.q_model.policy.evaluate_actions(self.q_model.policy.obs_to_tensor(self.q_env.get_obs())[0], torch.Tensor(np.array([action])).to("cuda"))
                        _, _, _, _, _ = self.q_env.step(action)
                        curr_qval += q_val

                    if curr_qval > max_qval:
                        max_qval = curr_qval
                        best_word = word
                else:
                    # otherwise return the word from eric heuristic
                    return word
        return best_word

    def get_vocab(self, vocab_file):
        vocab = []
        with open(vocab_file, 'r') as f:
            for l in f:
                vocab.append(l.strip())

        # Count letter frequencies at each index
        letter_freqs = [{letter: 0 for letter in string.ascii_lowercase} for _ in range(self.num_letters)]
        for word in vocab:
            for i, l in enumerate(word):
                letter_freqs[i][l] += 1

        # Assign a score to each letter at each index by the probability of it appearing
        letter_scores = [{letter: 0 for letter in string.ascii_lowercase} for _ in range(self.num_letters)]
        for i in range(len(letter_scores)):
            max_freq = np.max(list(letter_freqs[i].values()))
            for l in letter_scores[i].keys():
                letter_scores[i][l] = letter_freqs[i][l] / max_freq

        # Find a sorted list of words ranked by sum of letter scores
        vocab_scores = {}  # (score, word)
        for word in vocab:
            score = 0
            for i, l in enumerate(word):
                score += letter_scores[i][l]

            # # Optimization: If repeating letters, deduct a couple points
            # if len(set(word)) < len(word):
            #     score -= 0.25 * (len(word) - len(set(word)))

            vocab_scores[word] = score

        return vocab, vocab_scores, letter_scores
