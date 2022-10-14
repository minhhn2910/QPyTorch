#!/usr/bin/env python
# coding: utf-8

"""
CIFAR10 Low Precision Training Example in Floating Point 8 format
Training a Deep Neural Network (DNN) in low precision using Floating Point in QPyTorch
Follows configuration used in ["Training Deep Neural Networks with 8-bit Floating Point Numbers"](https://papers.nips.cc/paper/2018/file/335d3d1cd7ef05ec77714a215134914c-Paper.pdf)
Define a low-precision ResNet and recursively insert Quantizier() after every convolution layer
*Note: Quantization of weight, gradient, momentum, and gradient accumulator are not handled here.*
Use low-precision optimizer wrapper to define the quantization of weight, gradient, momentum, and gradient accumulator.
"""
# TODO: Parameterize bit-widths


# Imports
import argparse
import os
import sys
import math
from tqdm.notebook import trange, tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint, Posit
import wandb


def get_loaders():
    """
    Load CIFAR10 dataset and return train and test data loaders
    """
    ds = torchvision.datasets.CIFAR10
    path = os.path.join("./data", "CIFAR10")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = ds(path, train=True, download=True, transform=transform_train)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            test_set, batch_size=128, num_workers=4, pin_memory=True
        ),
    }
    return loaders


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreResNet(nn.Module):
    def __init__(self, quant_small, quant_large, num_classes=10, depth=20):
        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6

        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant_small)
        self.layer2 = self._make_layer(block, 32, n, quant_small, stride=2)
        self.layer3 = self._make_layer(block, 64, n, quant_small, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quant_small = quant_small()
        self.quant_large = quant_large()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_large(x)
        x = self.conv1(x)
        x = self.quant_small(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)
        x = self.quant_small(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.quant_large(x)

        return x


def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    ## Latency
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase == "train":
        model.train()
    elif phase == "eval":
        model.eval()
    ttl = 0

    start.record()
    with torch.autograd.set_grad_enabled(phase == "train"):
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)
            output = model(input)
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase == "train":
                loss = loss * 1000  # do loss scaling
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    end.record()
    torch.cuda.synchronize()

    correct = correct.cpu().item()
    return {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
        "time": start.elapsed_time(end),
    }


def parse_args(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-id", "--run_id", type=str, default=None, help="Run ID for logging"
    # )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "-a", "--arith", type=str, default="fp", help="Arithmetic type: fp or posit"
    )
    # parser.add_argument(
    #     "-e", "--exponent", type=int, default=None, help="Bit-width size of exponent"
    # )
    # parser.add_argument(
    #     "-m", "--mantissa", type=int, default=None, help="Bit-width size of mantissa"
    # )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training. Default is cuda:0. Set to 'cpu' to use cpu",
    )
    return parser.parse_args(args)


def main(args=None):
    # Args
    if not args:
        args = parse_args(sys.argv[1:])

    # Log
    wandb.init(project="qpy+", entity="glennmatlin", config=args)

    # Numerical Representation
    if args.arith == "fp":
        number_small = FloatingPoint(exp=5, man=2)
        number_large = FloatingPoint(exp=6, man=9)
    elif args.arith == "posit":
        number_small = Posit(nsize=8, es=2)
        number_large = Posit(nsize=16, es=2)
    else:
        raise ValueError("Invalid arithmetic type")

    # Quantizer
    weight_quant = quantizer(forward_number=number_small, forward_rounding="nearest")
    grad_quant = quantizer(forward_number=number_small, forward_rounding="nearest")
    momentum_quant = quantizer(
        forward_number=number_large, forward_rounding="stochastic"
    )
    acc_quant = quantizer(forward_number=number_large, forward_rounding="stochastic")

    # Lambda function to easily duplicate Quantizer()
    quant_small = lambda: Quantizer(
        forward_number=number_small,
        backward_number=number_small,
        forward_rounding="nearest",
        backward_rounding="nearest",
    )
    quant_large = lambda: Quantizer(
        forward_number=number_large,
        backward_number=number_large,
        forward_rounding="nearest",
        backward_rounding="nearest",
    )

    # Model
    model = PreResNet(quant_small, quant_large)
    if "cuda" not in args.device:
        model = model.cpu()
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.to(device=args.device)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    optimizer = OptimLP(
        optimizer,
        weight_quant=weight_quant,
        grad_quant=grad_quant,
        momentum_quant=momentum_quant,
        acc_quant=acc_quant,
        grad_scaling=1 / 1000,  # Do Loss Scaling
    )

    # Data
    loaders = get_loaders()

    # Train
    for epoch in trange(args.n_epochs):
        train_res = run_epoch(
            loaders["train"], model, F.cross_entropy, optimizer=optimizer, phase="train"
        )

        test_res = run_epoch(
            loaders["test"], model, F.cross_entropy, optimizer=optimizer, phase="eval"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "train_acc": train_res["accuracy"],
                "train_time": train_res["time"],
                "test_loss": test_res["loss"],
                "test_acc": test_res["accuracy"],
                "test_time": test_res["time"],
            }
        )
        # wandb.watch(model)


if __name__ == "__main__":
    main()
