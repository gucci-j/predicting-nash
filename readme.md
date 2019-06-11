Prediction of Nash Bargaining Solution in Negotiation Dialogue
===

This is the PyTorch implementation of [Prediction of Nash Bargaining Solution in Negotiation Dialogue](https://link.springer.com/chapter/10.1007/978-3-319-97304-3_60).

## Requirements
* PyTorch
* torchtext

## How to Run
You have to designate whether you conduct K-Fold CV or not with a json file.  
A sample json file is given by "param.json".

`python train.py param.json`


## Result
I iterated 20 epochs of training same as the paper with K-Fold CV (k=10). The result is shown as below.

```
06/08/2019 15:03:43 - INFO - __main__ -   LOSS: 0.0404475896836569, ACC: 0.7107541660467783, COR: 0.6553048329282829
```

The original paper reported that its model achieved an accuracy of 70 % for the prediction. This implementation also achieved the accuracy of 71 %.

## Note
Trained weights for the model is available, and stored in `./weight/weight.pth`.

## License
[MIT](./LICENSE)