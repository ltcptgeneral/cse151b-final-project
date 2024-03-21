import argparse
from ai import AI
import numpy as np
from tqdm import tqdm

global solution

def result_callback(word):

    global solution

    result = ['0', '0', '0', '0', '0']

    for i, letter in enumerate(word):

        if solution[i] == word[i]:
            result[i] = '2'
        elif letter in solution:
            result[i] = '1'
        else:
            pass

    return result

def main(args):
    global solution 

    if args.n is None:
        raise Exception('Need to specify n (i.e. n = 1 for wordle, n = 4 for quordle, n = 16 for sedecordle).')

    ai = AI(args.vocab_file, args.model_file, use_q_model=args.q_model)

    total_guesses = 0
    wins = 0
    num_eval = args.num_eval

    np.random.seed(0)

    for i in tqdm(range(num_eval)):
        idx = np.random.choice(range(len(ai.vocab)))
        solution = ai.vocab[idx]

        ai.reset(solution)

        guesses, word = ai.solve_eval(results_callback=result_callback)
        if word != solution:
            total_guesses += 5
        else:
            total_guesses += guesses
            wins += 1

    print(f"q_model?: {args.q_model} \t average guesses per game: {total_guesses / num_eval} \t win rate: {wins / num_eval}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, default=None)
    parser.add_argument('--vocab_file', dest='vocab_file', type=str, default='wordle_words.txt')
    parser.add_argument('--num_eval', dest="num_eval", type=int, default=1000)
    parser.add_argument('--model_file', dest="model_file", type=str, default='wordle_ppo_model')
    parser.add_argument('--q_model', dest="q_model", type=bool, default=False)
    args = parser.parse_args()
    main(args)