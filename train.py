from load_data import load_data

import torch.nn as nn
import torch.optim as optim
import torch

# from torchsummary import summary
from model import Model

# 交差分割検証とattentionの可視化
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
    torch.save(model.state_dict(), PATH)


def train_run(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        output = model(batch.text) # batch.text: (sentence length, batch_size)
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
            predictions = model(batch.text)
            label = batch_label_make(batch.value1, batch.value2, batch.value3) # label: (batch_size, output_dim)
            loss = criterion(predictions, label)
            acc = selection_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def batch_label_make(label1, label2, label3):
    return torch.cat([label1.unsqueeze(1), label2.unsqueeze(1), label3.unsqueeze(1)], dim=1)

def selection_accuracy(preds, y):
    """
    obtain the accuracy of the optimal essential issue prediction
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
