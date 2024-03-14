import contextlib
import pathlib
import random
from string import ascii_letters, ascii_lowercase

from rich.console import Console
from rich.theme import Theme

console = Console(width=40, theme=Theme({"warning": "red on yellow"}))

NUM_LETTERS = 5
NUM_GUESSES = 6
WORDS_PATH = pathlib.Path(__file__).parent / "wordlist.txt"


class Wordle:

    def __init__(self) -> None:
        self.word_list = WORDS_PATH.read_text(encoding="utf-8").split("\n")
        self.n_guesses = 6
        self.num_letters = 5
        self.curr_word = None
        self.reset()

    def refresh_page(self, headline):
        console.clear()
        console.rule(f"[bold blue]:leafy_green: {headline} :leafy_green:[/]\n")

    def start_game(self):
        # get a new random word
        word = self.get_random_word(self.word_list)

        self.curr_word = word

    def get_state(self):
        return 

    def action_to_word(self, action):
        # Calculate the word from the array
        word = ''
        for i in range(0, len(ascii_lowercase), 26):
            # Find the index of 1 in each block of 26
            letter_index = action[i:i+26].index(1)
            # Append the corresponding letter to the word
            word += ascii_lowercase[letter_index]

        return word

    def play_guess(self, action):
        # probably an array of length 26 * 5 for 26 letters and 5 positions
        guess = action

    def get_random_word(self, word_list):
        if words := [
            word.upper()
            for word in word_list
            if len(word) == NUM_LETTERS
            and all(letter in ascii_letters for letter in word)
        ]:
            return random.choice(words)
        else:
            console.print(
                f"No words of length {NUM_LETTERS} in the word list",
                style="warning",
            )
            raise SystemExit()

    def show_guesses(self, guesses, word):
        letter_status = {letter: letter for letter in ascii_lowercase}
        for guess in guesses:
            styled_guess = []
            for letter, correct in zip(guess, word):
                if letter == correct:
                    style = "bold white on green"
                elif letter in word:
                    style = "bold white on yellow"
                elif letter in ascii_letters:
                    style = "white on #666666"
                else:
                    style = "dim"
                styled_guess.append(f"[{style}]{letter}[/]")
                if letter != "_":
                    letter_status[letter] = f"[{style}]{letter}[/]"

            console.print("".join(styled_guess), justify="center")
        console.print("\n" + "".join(letter_status.values()), justify="center")

    def guess_word(self, previous_guesses):
        guess = console.input("\nGuess word: ").upper()

        if guess in previous_guesses:
            console.print(f"You've already guessed {guess}.", style="warning")
            return guess_word(previous_guesses)

        if len(guess) != NUM_LETTERS:
            console.print(
                f"Your guess must be {NUM_LETTERS} letters.", style="warning"
            )
            return guess_word(previous_guesses)

        if any((invalid := letter) not in ascii_letters for letter in guess):
            console.print(
                f"Invalid letter: '{invalid}'. Please use English letters.",
                style="warning",
            )
            return guess_word(previous_guesses)

        return guess

    def reset(self, guesses, word, guessed_correctly, n_episodes):
        refresh_page(headline=f"Game: {n_episodes}")

        if guessed_correctly:
            console.print(f"\n[bold white on green]Correct, the word is {word}[/]")
        else:
            console.print(f"\n[bold white on red]Sorry, the word was {word}[/]")

    if __name__ == "__main__":
        main()
