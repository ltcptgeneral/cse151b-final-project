import string

import numpy as np

words = []
with open('words.txt', 'r') as f:
    for l in f:
        words.append(l.strip())

# Count letter frequencies at each index
letter_freqs = [{letter: 0 for letter in string.ascii_lowercase} for _ in range(5)]
for word in words:
    for i, l in enumerate(word):
        letter_freqs[i][l] += 1

# Assign a score to each letter at each index by the probability of it appearing
letter_scores = [{letter: 0 for letter in string.ascii_lowercase} for _ in range(5)]
for i in range(len(letter_scores)):
    max_freq = np.max(list(letter_freqs[i].values()))
    for l in letter_scores[i].keys():
        letter_scores[i][l] = letter_freqs[i][l] / max_freq

# Find a sorted list of words ranked by sum of letter scores
word_scores = []  # (score, word)
for word in words:
    score = 0
    for i, l in enumerate(word):
        score += letter_scores[i][l]
    word_scores.append((score, word))

sorted_by_second = sorted(word_scores, key=lambda tup: tup[0])[::-1]
print(sorted_by_second[:10])

for i, (score, word) in enumerate(sorted_by_second):
    if word == 'soare':
        print(f'{word} with a score of {score} is found at index {i}')

