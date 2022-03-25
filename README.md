# Deep Residual Networks for Image Classification

Pytorch implementation of ResNet18 for CIFAR10 classification.

## Authors

It's a joint work between Aditya, Vijay and Yash

- [Aditya Wagh](https://www.github.com/adityamwagh)
- [Vijayraj Gohil](https://www.github.com/vraj130)
- [Yash Patel](https://www.github.com/yyashpatel)


## Setup

Install either of [Anaconda](https://www.anaconda.com/), [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge/releases/tag/4.11.0-4).
All of them provide the `conda` package manager for python which we will use to install our project packages.

The code has beed tested using the following python packages.

- `torch >= v1.10`
- `torchvision >= v0.11` 

Create a virtual environment to install project dependencies.

```bash
conda create -n torch
```

Activate the virtual environment.
```
conda activate torch
```

Install **PyTorch** and **Torchvision** using from the pytorch channel.
```
conda install pytorch torchvision matplotlib cudatoolkit=11.3 -c pytorch

```

The `main.py` script contains the training and testing code. It imports the model from `project1_model.py`. The script takes the following arguments. 
```
usage: main.py [-h] -en EXPERIMENT_NUMBER -o OPTIMISER [-d DEVICE] [-e EPOCHS] [-lr LEARNING_RATE] [-mo MOMENTUM] [-wd WEIGHT_DECAY] -dp DATA_PATH

optional arguments:
  -h, --help            show this help message and exit
  -en EXPERIMENT_NUMBER, --experiment_number EXPERIMENT_NUMBER
                        number to track the different experiments
  -o OPTIMISER, --optimiser OPTIMISER
                        optimizer for training
  -d DEVICE, --device DEVICE
                        device to train on, default is gpu
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train for, default is 120
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for the optimizer, default is 0.1
  -mo MOMENTUM, --momentum MOMENTUM
                        momentum value for optimizer if applicable, default is 0.9
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay value for the optimizer if applicable, default is 5e-4
  -dp DATA_PATH, --data-path DATA_PATH
                        path to the dataset
```

For example, we should run this script like this.

```
python main.py -en 0 -o sgd -d gpu -e 1 -dp data
```

The `test.py` scripts loads the model from the weights file `project1_model.pt`. I should be used to test on custom images.