import argparse
from ai import AI


def main(args):
    if args.n is None:
        raise Exception('Need to specify n (i.e. n = 1 for wordle, n = 4 for quordle, n = 16 for sedecordle).')

    ai = AI(args.vocab_file)
    ai.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, default=None)
    parser.add_argument('--vocab_file', dest='vocab_file', type=str, default='wordle_words.txt')
    args = parser.parse_args()
    main(args)