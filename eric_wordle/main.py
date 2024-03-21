import argparse
from ai import AI


def main(args):
    if args.n is None:
        raise Exception('Need to specify n (i.e. n = 1 for wordle, n = 4 for quordle, n = 16 for sedecordle).')
    print(f"using q model? {args.q_model}")
    ai = AI(args.vocab_file, args.model_file, use_q_model=args.q_model, device=args.device)
    ai.reset("lingo")
    ai.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, default=None)
    parser.add_argument('--vocab_file', dest='vocab_file', type=str, default='wordle_words.txt')
    parser.add_argument('--model_file', dest="model_file", type=str, default='wordle_ppo_model')
    parser.add_argument('--q_model', dest="q_model", type=bool, default=False)
    parser.add_argument('--device', dest="device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)