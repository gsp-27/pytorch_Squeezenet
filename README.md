### Pytorch Squeeznet

Pytorch implementation of Squeezenet model as described in https://arxiv.org/abs/1602.07360 on cifar-10 Data.

The definition of Squeezenet model is present **model.py**.
The training procedure resides in the file **main.py**

Command to train the Squeezenet model on CIFAR 10 data is:
```bash
python main.py --batch-size 32 --epoch 10
```
Other options which can be used are specified in **main.py**
Eg: if you want to use a pretrained_model
```bash
python main.py --batch-size 32 --epoch 10 --model_name "pretrained model"
```

I am currently using SGD for training : learning rate and weight decay are currently updated using a 55 epoch learning rule, this usually gives good performance, but if you want to use something of your own, you can specify it by passing **learning_rate** and **weight_decay** parameter like so

```bash
python main.py --batch-size 32 --epoch 10 --learning_rate 1e-3 --epoch_55
```
