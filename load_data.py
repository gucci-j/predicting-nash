import torchtext
from torchtext import data
from torchtext.vocab import GloVe
import torch
from pathlib import Path

from load_negotiation import load_negotiation
from model import Model

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

def load_data_cv(batch_size=16):
    TEXT = data.Field(sequential=True)
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    fields = [('text', TEXT), ('value1', LABEL), ('value2', LABEL), ('value3', LABEL)]

    _data = data.TabularDataset(
        path='./all.csv',
        format='csv',
        skip_header=True,
        fields=fields)
    
    kf = KFold(n_splits=10, random_state=1234)
    data_arr = np.array(_data.examples)

    for train_index, val_index in kf.split(data_arr):
        yield(
            TEXT,
            data.Dataset(data_arr[train_index], fields),
            data.Dataset(data_arr[val_index], fields),
        )


def load_data(batch_size=16, path='data/negotiate'):
    r_path = Path(path)
    for child_name in ['train.txt', 'test.txt', 'val.txt']:
        child_path = r_path / child_name
        load_negotiation(child_path)

    TEXT = data.Field(sequential=True)
    # sequential: whether input data is variable?
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

    train, val, test = data.TabularDataset.splits(
        path='./',
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        skip_header=True,
        fields=[('text', TEXT), ('value1', LABEL), ('value2', LABEL), ('value3', LABEL)])

    """
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))
    """

    TEXT.build_vocab(train, vectors="glove.6B.300d", min_freq=2)
    print(f'Embedding size: {TEXT.vocab.vectors.size()}.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), 
        batch_size=batch_size, 
        device=device,
        sort_key=lambda x: len(x.text), # needs to be told what function it should use to group the data.
        sort_within_batch=False)

    """
    batch = next(iter(train_iter))
    print(batch.text)
    print(batch.text.size())
    print(batch.value1)
    print(batch.value1.size())
    """

    return TEXT, LABEL, train_iter, val_iter, test_iter

if __name__ == '__main__':
    # load_data()
    load_data_cv()