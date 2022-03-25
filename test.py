import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( num_channels[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Resnet18(num_of_block, num_of_channel):
    return ResNet(BasicBlock, num_of_block, num_of_channel)


def test_model(test_loader, model, loss_fn, device):
     
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

    epoch_test_loss = running_test_loss / len(test_loader)
    epoch_test_accuracy = 100.0 * correct / total
    
    print(f"Test Accuracy: {epoch_test_accuracy}.3f %| Test Loss: {epoch_test_loss}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, required=False, default="cpu", help="device to train AND/OR test on, default is cpu")
    parser.add_argument("-dp", "--data-path", type=str, required=True, help="path to the dataset")
    # Parser for the Weight path
    parser.add_argument("-w", "--weights-path", type=str, required=True, default=None, help="path to the weights file")

    args = parser.parse_args()

    data_path = args.data_path
    device = args.device
    weight_path = args.weights_path

    # device to test on
    if args.device == "cpu" and torch.cuda.is_available() == True:
        device = "cpu"
    else:
        device = "cuda"

    model = Resnet18([2, 2, 2, 2], [42, 84, 168, 336])
    ## load weights in the model

    model.load_state_dict(torch.load(os.path.join(weight_path, "weight_best.pth")))

    model = model.to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_paramenters = count_parameters(model)
    print(model_paramenters)


    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)


    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()
    test_model(test_dataloader, model, loss_fn, device)