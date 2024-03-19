import pandas

print('Loading in words dictionary; this may take a while...')
df = pandas.read_json('words_dictionary.json')
print('Done loading words dictionary.')
words = []
for word in df.axes[0].tolist():
    if len(word) != 5:
        continue
    words.append(word)
words.sort()

with open('words.txt', 'w') as f:
    for word in words:
        f.write(word + '\n')
