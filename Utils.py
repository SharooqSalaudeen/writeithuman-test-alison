import argparse
from itertools import chain
import warnings
from tqdm import tqdm
from functools import partial
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import heapq
from collections import Counter
from random import random
import gc
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import pandas as pd
import datetime

import copy
import csv
import os
import time
import pickle
import itertools

from pandas import DataFrame

import spacy
from nltk import skipgrams
from nltk.util import ngrams
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model for fast POS tagging
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Increase max_length for large text processing (safe since parser/NER are disabled)
nlp.max_length = 20000000  # 20 million characters

tqdm = partial(tqdm, position=0, leave=True)

warnings.filterwarnings("ignore")


# Universal POS tags mapping (spaCy)
tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
idx = 0
temp = dict()
offset = 65
for tag in tags:
    if offset + idx == 90:
        offset = 72
    temp[tag] = chr(offset + idx)
    idx += 1
tags = temp


def to_char(x): return tags[x] if x in tags else x


def token_tag_join(text):
    doc = nlp(text)
    return ''.join([to_char(token.pos_) for token in doc])


def tag(texts):
    # Use spaCy's pipe for parallel processing
    docs = nlp.pipe(texts, batch_size=50, n_process=4)
    return [''.join([to_char(token.pos_) for token in doc]) for doc in docs]


def countSkip(skipgram, texts):

    total = 0
    m = len(skipgram)

    for text in texts:

        n = len(text)

        mat = [[0 for i in range(n + 1)] for j in range(m + 1)]
        for j in range(n + 1):
            mat[0][j] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                mat[i][j] = mat[i][j - 1]

                if skipgram[i - 1] == text[j - 1]:
                    mat[i][j] += mat[i - 1][j - 1]

        total += mat[m][n]

    return total


def get_skipgrams(text, n, k):
    if n > 1:
        ans = [skipgram for skipgram in skipgrams(text, n, k)]
    else:
        ans = ngrams(text, n)
    return ans


def return_best_pos_n_grams(n, L, pos_texts):
    n_grams = ngrams(pos_texts, n)

    data = dict(Counter(n_grams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams


def return_best_word_n_grams(n, L, tokens):

    all_ngrams = ngrams(tokens, n)

    data = dict(Counter(all_ngrams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams


def return_best_n_grams(n, L, text):

    n_grams = ngrams(text, n)

    data = dict(Counter(n_grams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams


def ngram_rep(text, pos_text, features):

    to_ret = []
    ret_idx = 0

    for idx in range(len(features[0])):
        num_ngrams = len(Counter(ngrams(text, len(features[0][idx][0]))))

        for n_gram in features[0][idx]:
            to_ret.append(text.count(''.join(n_gram)) /
                          num_ngrams if num_ngrams != 0 else 0)

    for idx in range(len(features[1])):
        num_pos_ngrams = len(
            Counter(ngrams(pos_text, len(features[1][idx][0]))))

        for pos_n_gram in features[1][idx]:
            to_ret.append(pos_text.count(''.join(pos_n_gram)) /
                          num_pos_ngrams if num_pos_ngrams != 0 else 0)

    words = tokenize(text)
    spaced_text = ' '.join(words)
    for idx in range(len(features[2])):
        num_word_ngrams = len(Counter(ngrams(words, len(features[2][idx][0]))))

        for word_ngram in features[2][idx]:
            to_ret.append(spaced_text.count(' '.join(word_ngram)) /
                          num_word_ngrams if num_word_ngrams != 0 else 0)

    return to_ret


def tokenize(text, show_progress=False):
    """Tokenize text using spaCy. For very large texts, this may take time."""
    if show_progress and len(text) > 1000000:
        print(f'  Processing {len(text):,} characters...')
    doc = nlp(text)
    if show_progress and len(text) > 1000000:
        print(f'  Generated {len(doc):,} tokens')
    return [token.text for token in doc]
