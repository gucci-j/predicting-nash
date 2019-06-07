from load_data import load_data

import torch.nn as nn
import torch.optim as optim
import torch

from model import Model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 交差分割検証
# 評価指標の実装: rank / nash
# test/dev用のメソッドも修正する
# init_hidden系の問題について考える
# attentionのマスクをどうするか
# early stoppingの導入

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
        valid_loss, valid_acc = eval_run(model, valid_iterator, criterion)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    
    test_loss, test_acc = eval_run(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
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
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions, _ = model(batch.text)
            label = batch_label_make(batch.value1, batch.value2, batch.value3) # label: (batch_size, output_dim)
            loss = criterion(predictions, label)
            acc = selection_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

# 順位係数もここに用意する

if __name__ == '__main__':
    train()
