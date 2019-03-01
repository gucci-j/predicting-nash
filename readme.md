Prediction of Nash Bargaining Solution in Negotiation Dialogue
===

This is the PyTorch implementation of [Prediction of Nash Bargaining Solution in Negotiation Dialogue](https://link.springer.com/chapter/10.1007/978-3-319-97304-3_60).

As of now, I have only implemented the feature of predicting an optimal essential issue. I will update this repository, once the implementation of another feature is done.

## Requirements
* PyTorch
* torchtext

## How to Run
`python train.py`

## Result
I iterated 20 epochs of training same as the paper. The result is shown as below.

```
| Epoch: 01 | Train Loss: 0.074 | Train Acc: 48.99% | Val. Loss: 0.100 | Val. Acc: 46.95% |
| Epoch: 02 | Train Loss: 0.046 | Train Acc: 66.74% | Val. Loss: 0.047 | Val. Acc: 67.01% |
| Epoch: 03 | Train Loss: 0.039 | Train Acc: 71.25% | Val. Loss: 0.041 | Val. Acc: 72.88% |
| Epoch: 04 | Train Loss: 0.035 | Train Acc: 72.91% | Val. Loss: 0.046 | Val. Acc: 67.79% |
| Epoch: 05 | Train Loss: 0.032 | Train Acc: 74.74% | Val. Loss: 0.035 | Val. Acc: 76.97% |
| Epoch: 06 | Train Loss: 0.030 | Train Acc: 75.60% | Val. Loss: 0.036 | Val. Acc: 73.84% |
| Epoch: 07 | Train Loss: 0.028 | Train Acc: 77.33% | Val. Loss: 0.038 | Val. Acc: 73.73% |
| Epoch: 08 | Train Loss: 0.026 | Train Acc: 78.19% | Val. Loss: 0.036 | Val. Acc: 72.07% |
| Epoch: 09 | Train Loss: 0.024 | Train Acc: 79.03% | Val. Loss: 0.036 | Val. Acc: 72.84% |
| Epoch: 10 | Train Loss: 0.022 | Train Acc: 80.17% | Val. Loss: 0.036 | Val. Acc: 73.42% |
| Epoch: 11 | Train Loss: 0.021 | Train Acc: 81.27% | Val. Loss: 0.034 | Val. Acc: 74.81% |
| Epoch: 12 | Train Loss: 0.019 | Train Acc: 82.67% | Val. Loss: 0.036 | Val. Acc: 72.96% |
| Epoch: 13 | Train Loss: 0.018 | Train Acc: 83.08% | Val. Loss: 0.038 | Val. Acc: 71.95% |
| Epoch: 14 | Train Loss: 0.016 | Train Acc: 84.06% | Val. Loss: 0.041 | Val. Acc: 73.69% |
| Epoch: 15 | Train Loss: 0.016 | Train Acc: 83.89% | Val. Loss: 0.041 | Val. Acc: 73.84% |
| Epoch: 16 | Train Loss: 0.015 | Train Acc: 84.90% | Val. Loss: 0.040 | Val. Acc: 72.88% |
| Epoch: 17 | Train Loss: 0.014 | Train Acc: 84.77% | Val. Loss: 0.038 | Val. Acc: 72.72% |
| Epoch: 18 | Train Loss: 0.013 | Train Acc: 85.70% | Val. Loss: 0.037 | Val. Acc: 74.23% |
| Epoch: 19 | Train Loss: 0.013 | Train Acc: 85.45% | Val. Loss: 0.038 | Val. Acc: 74.34% |
| Epoch: 20 | Train Loss: 0.012 | Train Acc: 86.52% | Val. Loss: 0.039 | Val. Acc: 72.92% |
| Test Loss: 0.039 | Test Acc: 70.79% |
```

The paper reported that its model achieved an accuracy of 70 % for the prediction. This implementation also achieved the accuracy of 70.79 % without cross validation.

## Note
Trained weights for the model is available, and stored in `./weight/weight.pth`.

## License
[MIT](./LICENSE)