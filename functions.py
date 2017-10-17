
import pandas as pd
import numpy as np
import re
import string
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer


def cleanup_str(st, numbers=False):
    if type(st) == bytes:
        try:
            st = st.decode('utf-8').strip().lower()
        except:
            print('unicode error: {}'.format(st))

    if numbers == True:
        keep = set(string.ascii_lowercase + string.digits + string.punctuation + ' ')
    else:
        keep = set(string.ascii_lowercase + string.punctuation + ' ')

    # clean string
    st = ''.join(x if x in keep else ' ' for x in st)
    # rem multiple spaces
    st = re.sub(' +', ' ', st)

    return st


# mapper: cleanup a pd column or list of strings
def cleanup_col(col, numbers=False):
    col = map(lambda x: cleanup_str(x, numbers=numbers), col)
    return list(col)

def binarize_tokenized(X, vocab_len):
    binarizer = LabelBinarizer()
    binarizer.fit(range(vocab_len))
    X = np.array([binarizer.transform(x) for x in X])

    return X


def char_preproc(X, Y, vocab_len, binarize=False):
    # -----------------------------
    # preproc X's------------------

    # cleanup
    X = cleanup_col(X, numbers=True)
    # split in arrays of characters
    char_arrs = [[x for x in y] for y in X]

    # tokenize
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(char_arrs)

    # token sequences
    seq = tokenizer.texts_to_sequences(X)

    # pad to same length
    seq = pad_sequences(seq, maxlen=250, padding='post', truncating='post', value=0)

    # make to on-hot
    if binarize:
        X = binarize_tokenized(seq, vocab_len)
    else:
        X = seq

    # ----------------------------
    # preproce Y's and return data

    # one-hot encode Y's
    Y = np.array([[1, 0] if x == 1 else [0, 1] for x in Y])

    # generate and return final dataset
    data = Dataset(X, Y, shuffle=True, testsize=0.1)

    return data



def load_processed_data(load=True, binarize=False):
    table = None

    if os.path.isfile('data/processed/data-ready.pkl') and load:
        print("data exists - loading")

        with open('data/processed/data-ready.pkl', 'rb') as file:
            data = pickle.load(file)
    else:
        print("reading raw data and preprocessing..")
        table = pd.read_csv('data/rt-polarity.csv')
        data = char_preproc(table.text, table.label, 70, binarize)

        with open('data/processed/data-ready.pkl', 'wb') as file:
            pickle.dump(data, file)

    return (data, table)


class Dataset():
    def __init__(self, x, y=None, testsize=0.2, shuffle=False):

        lend = len(x)

        if testsize == None:
            self.x_data = x
            if y != None:
                self.y_data = y

            print('Single dataset of size {}'.format(lend))
        else:
            if shuffle:
                si = np.random.permutation(np.arange(lend))
                x = x[si]
                y = y[si]
                self.si = si

            if type(testsize) == int:
                testindex = testsize
            else:
                testindex = int(testsize * lend)

            self.x_train = x[testindex:]
            self.x_test = x[:testindex]
            self.y_train = y[testindex:]
            self.y_test = y[:testindex]
            self.testindex = testindex

            print('Train size: {}, test size {}'.format(len(self.y_train), len(self.y_test)))

