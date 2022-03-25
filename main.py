import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from project1_model import project1_model


# Defining the Training Loop
def train_test_model(epochs, train_loader, test_loader, model, loss_fn, optimizer):

    for epoch in range(1, epochs + 1):
        """
        Model training
        """
        model.train()
        running_train_loss = 0
        correct = 0
        total = 0

# Scheduling the Learning Rate manually
        if(epoch>90 and epoch<105):
            optimizer.param_groups[0]['lr'] = 0.01
        elif(epoch>105 and epoch<120):
            optimizer.param_groups[0]['lr'] = 0.001

        print("Epoch: {}".format(epoch))

        for imgs, labels in train_loader:

            X, y = imgs.to(device), labels.to(device)
            train_pred = model(X)
            train_loss = loss_fn(train_pred, y)

            # Back prop
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
            _, predicted = train_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_loss_values.append(epoch_train_loss)
        epoch_train_accuracy = 100.0 * correct / total
        train_accuracy_values.append(epoch_train_accuracy)

        # print accuracy and loss at every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss} | Train Accuracy: {epoch_train_accuracy} %")

        """
        Model testing
        """
        model.eval()
        running_test_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in test_loader:

            X, y = imgs.to(device), labels.to(device)
            pred = model(X)

            # compute test Loss
            loss = loss_fn(pred, y)
            running_test_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)

            # correct += (predicted == labels).sum().item()
            correct += predicted.eq(labels.to(device)).sum().item()

        epoch_test_loss = running_test_loss / len(test_dataloader)
        test_loss_values.append(epoch_test_loss)
        epoch_test_accuracy = 100.0 * correct / total
        test_accuracy_values.append(epoch_test_accuracy)

        # print accuracy and loss at every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Test Loss: {epoch_test_loss} | Test Accuracy: {epoch_test_accuracy} %")

def select_optimiser(argument, model):

    # stochastic gradient descent
    if argument == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # stochastic gradient descent with nesterov correction
    if argument == "sgd_nest":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # adaptive gradient descent
    if argument == "adagrad":
        return optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # adadelta
    if argument == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # adam optimizer
    if argument == "adam":
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)


if __name__ == "__main__":

    """
    Provision for arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-en", "--experiment_number", type=str, required=True, help="number to track the different experiments")
    parser.add_argument("-o", "--optimiser", type=str, required=True, help="optimizer for training")
    # parser.add_argument("-m", "--model", type=str, required=False, default = "Resnet", help="model to be used")
    parser.add_argument("-d", "--device", type=str, required=False, default="gpu", help="device to train on, default is gpu")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=120, help="number of epochs to train for, default is 120")
    parser.add_argument("-lr", "--learning-rate", type=float, required=False, default=0.1, help="learning rate for the optimizer, default is 0.1")
    parser.add_argument("-mo", "--momentum", type=float, required=False, default=0.9, help="momentum value for optimizer if applicable, default is 0.9")
    parser.add_argument("-wd", "--weight-decay", type=float, required=False, default=5e-4, help="weight decay value for the optimizer if applicable, default is 5e-4")
    parser.add_argument("-dp", "--data-path", type=str, required=True, help="path to the dataset")
    # parser.add_argument("-b", "--blocks", nargs=4, required=True, type=int, help="number of blocks in each layer")
    # parser.add_argument("-c", "--channels", nargs=4, required=True, type=int, help="number of channels in each layer")

    args = parser.parse_args()

    """
    Hyperparameters
    """

    # path to the dataset
    data_path = args.data_path

    # number of epochs
    epochs = args.epochs

    # block size for resnet
    # blocks = args.blocks

    # number of convolutional filters in each conv layer
    # channels = args.channels

    # device to train on
    if args.device == "gpu" and torch.cuda.is_available() == True:
        device = "cuda"
    else:
        device = "cpu"

    # loss function initialization
    loss = nn.CrossEntropyLoss()

    # model initialization
    # resnet_model = project1_model(blocks, channels).to(device)
    resnet_model = project1_model().to(device)

    optimizer = select_optimiser(args.optimiser, resnet_model)

    """
    Data Related Stuff
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_transforms2 = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomCrop((32,32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    train_transforms3 = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomCrop((32,32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_data = torchvision.datasets.CIFAR10(data_path, train=True, transform=train_transforms3, download=True)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=True)

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    """
    Model Training And Evaluation
    """
    print("Started Training\n")

    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    train_test_model(epochs, train_dataloader, test_dataloader, resnet_model, loss, optimizer)

    print("Finished Training\n")

    # save the training & testing loss & accuracy to the disk
    os.makedirs(os.path.join(os.getcwd(), "metrics"), exist_ok=True)
    np.save(os.path.join("metrics", args.experiment_number + "_train_loss.npy"), train_loss_values)
    np.save(os.path.join("metrics", args.experiment_number + "_test_loss.npy"), test_loss_values)
    np.save(os.path.join("metrics", args.experiment_number + "_train_accuracy.npy"), train_accuracy_values)
    np.save(os.path.join("metrics", args.experiment_number + "_test_accuracy.npy"), test_accuracy_values)

    #  save model weights to weights folder
    os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)
    PATH_MODEL_WEIGHTS = os.path.join("weights", args.experiment_number + "_weights.pth")
    torch.save(resnet_model.state_dict(), PATH_MODEL_WEIGHTS)
    print(f"Saved model weights to {PATH_MODEL_WEIGHTS}")

    # print final results and settings 
    print(f"Final Training Accuracy: {train_accuracy_values[-1]: .3f} % | Final Test Accuracy: {test_accuracy_values[-1]: .3f} %\n")
    print("This accuracy is achieved with the following Settings")
    print(f"Number of Blocks in each layer: {blocks}\n Number of Channels in each layer: {channels}")
    print(f"Optimiser: {args.optimiser}\n Learning Rate: {args.learning_rate}")