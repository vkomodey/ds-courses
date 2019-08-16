import matplotlib
import numpy as np
import pandas as pd
import Levenshtein as Lev
import re

pd.set_option('display.max_rows', 1000)

tokens = None
with open('./word-frequency/article', 'r') as f:
    text = f.read().strip()


# TODO: upgrade splitting. now empty values are in list
tokens = re.split(r'[\n\r\s\.,\'\"\(\):]+', text)

frequency_map = {}

def are_similar(w1, w2):
    max_length = max(len(w1), len(w2))

    if Lev.distance(w1, w2) < max_length // 2:
        return True

    return False


for token in tokens:
    if len(token) <= 3 or token.isnumeric():
        continue

    if token in frequency_map.keys():
        frequency_map[token]['count'] = frequency_map[token]['count'] + 1
    else:
        frequency_map[token] = {
            'count': 1,
            'similar_to': []
        }

    for f_word in frequency_map.keys():
        if f_word is token:
            continue

        if are_similar(token, f_word):
            frequency_map[f_word]['count'] += 1
            frequency_map[f_word]['similar_to'].append(token)


df = pd.DataFrame({
    "Frequency": list(map(lambda f: f['count'], frequency_map.values())),
    "Similar": list(map(lambda f: f['similar_to'], frequency_map.values()))
}, index=frequency_map.keys())

# df.sort_values(by=['Frequency'], ascending=False).hist()
df.Frequency.hist()
