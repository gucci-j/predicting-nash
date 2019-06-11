from load_data import load_data, load_data_cv

import torch.nn as nn
import torch.optim as optim
import torch
from torchtext import data

from model import Model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import sys
from pathlib import Path
import json

#
# Loading Arguments
#
if len(sys.argv) <= 1:
    raise Exception('Please give json settings file path!')
args_p = Path(sys.argv[1])
if args_p.exists() is False:
    raise Exception('Path not found. Please check an argument again!')

with args_p.open(mode='r') as f:
    true = True
    false = False
    null = None
    args = json.load(f)


#
# Logging
# Reference: https://qiita.com/knknkn1162/items/87b1153c212b27bd52b4
#
import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

# for cross valiadtion
def train_cv():
    # hyperparameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    NUM_LAYERS = 3
    INPUT_DROPOUT = 0.2
    HIDDEN_DROPOUT = 0.5
    N_EPOCHS = 20

    _history = []

    for TEXT, train_data, val_data in load_data_cv():
        TEXT.build_vocab(train_data, vectors="glove.6B.300d", min_freq=2)
        INPUT_DIM = len(TEXT.vocab)
        logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_iterator = data.Iterator(train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), device=device)
        val_iterator = data.Iterator(val_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), device=device)

        model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, 
                INPUT_DROPOUT, HIDDEN_DROPOUT, TEXT)
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-07)
        criterion = nn.MSELoss()

        # for a gpu environment
        model = model.to(device)
        criterion = criterion.to(device)

        for epoch in range(N_EPOCHS):
            train_loss, train_acc  = train_run(model, train_iterator, optimizer, criterion)
            logger.info(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        
        val_loss, val_acc, val_cor = eval_run(model, val_iterator, criterion)
        logger.info(f'| Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f} | Val Cor: {val_cor:.3f}% |')
        _history.append([val_loss, val_acc, val_cor])
    
    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean(_history[:, 1])
    cor = np.mean(_history[:, 2])
    
    logger.info(f'LOSS: {loss}, ACC: {acc}, COR: {cor}')

def train():
    # hyperparameters
    BATCH_SIZE = 32
    TEXT, LABEL, train_iterator, valid_iterator, test_iterator = load_data(batch_size=BATCH_SIZE)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    NUM_LAYERS = 3
    INPUT_DROPOUT = 0.2
    HIDDEN_DROPOUT = 0.5
    N_EPOCHS = 20
    PATH = './weight/weight.pth'

    model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, 
            INPUT_DROPOUT, HIDDEN_DROPOUT, TEXT)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-07)
    criterion = nn.MSELoss()

    # for a gpu environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    # print summary
    # summary(model, (100, BATCH_SIZE))

    for epoch in range(N_EPOCHS):
        train_loss, train_acc  = train_run(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, val_cor = eval_run(model, valid_iterator, criterion)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f} | Val. Cor: {val_cor:.3f}% |')
    
    test_loss, test_acc, test_cor = eval_run(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f} | Test Cor: {test_cor:.3f}% |')
    attn_visualization(model, test_iterator, TEXT, multiple_flag=True)
    torch.save(model.state_dict(), PATH)


def train_run(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        output, _ = model(batch.text) # batch.text: (sentence length, batch_size)
        label = batch_label_make(batch.value1, batch.value2, batch.value3) # label: (batch_size, output_dim)
        # print(f'{output.size()}, {label.size()}')

        loss = criterion(output, label)
        acc = selection_accuracy(output, label)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_run(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_cor = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions, _ = model(batch.text)
            label = batch_label_make(batch.value1, batch.value2, batch.value3) # label: (batch_size, output_dim)
            loss = criterion(predictions, label)
            acc = selection_accuracy(predictions, label)
            cor = spearman_correlation(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_cor += cor.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_cor / len(iterator)

def attn_visualization(model, iterator, TEXT, multiple_flag=False):
    """
    Visualize self-attention weights with input captions.
    """

    if multiple_flag is False:
        with torch.no_grad():
            batch = next(iter(iterator))
            _, attention = model(batch.text)

            # in torchtext, batch_size is placed in dim=1. dim=0 is used for sentence length
            text = batch.text.transpose(0, 1)
            # print(attention.size())
            attention_weight = attention.cpu().numpy()

            itos = []
            for text_element in text:
                itos_element = []
                for index in text_element:
                    # print(f'{TEXT.vocab.itos[index]} ')
                    itos_element.append(TEXT.vocab.itos[index])
                itos.append(itos_element)

            plt.figure(figsize = (16, 5))
            sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
            plt.savefig('attention.png')

    elif multiple_flag is not False:
        with torch.no_grad():
            batch_count = 0
            for batch in iterator:
                _, attention = model(batch.text)
                text = batch.text.transpose(0, 1)
                attention_weight = attention.cpu().numpy()
                
                itos = []
                for text_element in text:
                    itos_element = []
                    for index in text_element:
                        itos_element.append(TEXT.vocab.itos[index])
                    itos.append(itos_element)
                
                fig_size = len(batch.text) + 1 # for changing fig_size dynamically
                plt.figure(figsize = (fig_size, 7))
                sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
                plt.savefig('./fig/attention_' + str(batch_count) + '.png')
                plt.close()
                batch_count += 1

def batch_label_make(label1, label2, label3):
    return torch.cat([label1.unsqueeze(1), label2.unsqueeze(1), label3.unsqueeze(1)], dim=1)

def selection_accuracy(preds, y):
    """
    Obtain the accuracy of the optimal essential issue prediction.
    """

    # get indices of each max value: preds and y
    preds_max_index = preds.argmax(dim=1)
    y_max_index = y.argmax(dim=1)
    correct = (preds_max_index == y_max_index).sum(dim=0).float()
    acc = correct / len(preds)

    return acc

#
# 順位相関をここで見る
# 参考: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
#
def spearman_correlation(preds, y):
    spearmanr_list = []
    _preds = preds.cpu().numpy()
    _y = y.cpu().numpy()
    for index, pred in enumerate(_preds):
        if _y[index][0] == _y[index][1] == _y[index][2]:
            # print(f'{_y[index][:3]},  {pred[:3]}')
            continue
        else:
            spearmanr_element = spearmanr(_y[index][:3], pred[:3]).correlation
            if not np.isnan(spearmanr_element):
                spearmanr_list.append(spearmanr_element)
    cor = sum(spearmanr_list) / len(spearmanr_list)
    return cor

if __name__ == '__main__':
    if args['cv'] is True:
        train_cv()
    else:
        train()
