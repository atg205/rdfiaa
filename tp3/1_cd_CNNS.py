import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *

PRINT_INTERVAL = 200
PATH="datasets"

class ConvNet(nn.Module):
    """
    This class defines the structure of the neural network
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        # We first define the convolution and pooling layers as a features extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(32, 64, (5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(64, 64, (5, 5), stride=1, padding=0),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        # We then define fully connected layers as a classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
            nn.ReLU(),
            # Reminder: The softmax is included in the loss, do not put it here
        )

    # Method called when we apply the network to an input batch
    def forward(self, input):
        bsize = input.size(0) # batch size
        output = self.features(input) # output of the conv layers
        output = output.view(bsize, -1) # we flatten the 2D feature maps into one 1D vector for each input
        output = self.classifier(output) # we compute the output of the fc layers
        return output



def get_dataset(batch_size, cuda=False):
    """
    This function loads the dataset and performs transformations on each
    image (listed in `transform = ...`).
    """
    train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=cuda, num_workers=2)

    return train_loader, val_loader



import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

PRINT_INTERVAL = 50


def epoch(data, model, criterion, optimizer=None, cuda=False):
    """
    Przebieg (train lub eval) na zbiorze `data`.
    """
    model.train() if optimizer else model.eval()

    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_top5_acc = AverageMeter()
    avg_batch_time = AverageMeter()

    tic = time.time()
    for i, (inputs, targets) in enumerate(data):

        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        batch_time = time.time() - tic
        tic = time.time()

        avg_loss.update(loss.item(), inputs.size(0))
        avg_top1_acc.update(prec1.item(), inputs.size(0))
        avg_top5_acc.update(prec5.item(), inputs.size(0))
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                  'Prec@5 {top5.val:5.1f} ({top5.avg:5.1f})'.format(
                   "EVAL" if optimizer is None else "TRAIN", i, len(data),
                   batch_time=avg_batch_time, loss=avg_loss,
                   top1=avg_top1_acc, top5=avg_top5_acc))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f}%\t'
          'Avg Prec@5 {top5.avg:5.2f}%\n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss,
           top1=avg_top1_acc, top5=avg_top5_acc))

    return avg_top1_acc.avg, avg_top5_acc.avg, avg_loss.avg


def main(batch_size=128, lr=0.1, epochs=5, cuda=False):
    # Utworzenie modelu, kryterium, optymalizatora
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if cuda:
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    # Dane
    train_loader, test_loader = get_dataset(batch_size, cuda)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/batch{batch_size}_lr{lr}")

    for epoch_idx in range(epochs):
        print(f"\n===== EPOCH {epoch_idx + 1}/{epochs} =====")

        # Train phase
        train_top1, train_top5, train_loss = epoch(train_loader, model, criterion, optimizer, cuda)

        # Eval phase
        test_top1, test_top5, test_loss = epoch(test_loader, model, criterion, cuda=cuda)

        # Logowanie do TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch_idx)
        writer.add_scalar('Loss/test', test_loss, epoch_idx)
        writer.add_scalar('Accuracy/train_top1', train_top1, epoch_idx)
        writer.add_scalar('Accuracy/test_top1', test_top1, epoch_idx)

    writer.close()


if __name__ == "__main__":
    for batch_size in [64, 128, 256]:
        for lr in [1e-3, 1e-2, 1e-1]:
            main(batch_size=batch_size, lr=lr, cuda=False)
