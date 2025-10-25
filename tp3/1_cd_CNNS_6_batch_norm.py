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
import pandas as pd
from collections import defaultdict

PRINT_INTERVAL = 200
PATH="datasets"

class ConvNet(nn.Module):
    """
    This class defines the structure of the neural network
    """

    def __init__(self,dropout_rate=0.5,batch_norm = False):
        super(ConvNet, self).__init__()
        # We first define the convolution and pooling layers as a features extractor
        if batch_norm:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=2, padding=0),
                nn.Conv2d(32, 64, (5, 5), stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=2, padding=0),
                nn.Conv2d(64, 64, (5, 5), stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.MaxPool2d((2, 2), stride=2, padding=0,ceil_mode=True),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=2, padding=0),
                nn.Conv2d(32, 64, (5, 5), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=2, padding=0),
                nn.Conv2d(64, 64, (5, 5), stride=1, padding=0),
                nn.MaxPool2d((2, 2), stride=2, padding=0,ceil_mode=True),
            )
        # We then define fully connected layers as a classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 1000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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



def get_dataset(batch_size, cuda=False, crop =False,normalization=False):
    """
    This function loads the dataset and performs transformations on each
    image (listed in `transform = ...`).
    """
    mu = [0.491,0.482,0.447]
    sigma = [0.202,0.199,0.201]
    norm_func = transforms.Normalize(mean=mu, std=sigma) if normalization else torch.nn.Identity()
    if crop:
        train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28),
                transforms.ToTensor(),
                norm_func,

            ]))
        val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
            transform=transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                norm_func,

            ]))
    else:
        train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                norm_func,

            ]))
        val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                norm_func,

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


def epoch(data, model, criterion,optimizer=None, cuda=False,dropout_rate = 0.5,epoch_idx= 0):
    """
    Przebieg (train lub eval) na zbiorze `data`.
    """
    model.train() if optimizer else model.eval()
    writer_str = "train" if optimizer else "test"
    
    global_runs = len(data) * epoch_idx
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


        if optimizer:
            loss_plot.update(avg_loss.val)
        
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
    loss_plot.plot()
    loss_plot.save()

    return avg_top1_acc.avg, avg_top5_acc.avg, avg_loss.avg


def main(batch_size=128, lr=0.1, epochs=5, cuda=torch.cuda.is_available(),dropout_rate = 0.0,normalization=False,crop=False,lr_decay =1.0,batch_norm =False):
    # Utworzenie modelu, kryterium, optymalizatora
    model = ConvNet(dropout_rate=dropout_rate,batch_norm=batch_norm)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay)
    if cuda:
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    global loss_plot

    # Dane
    train_loader, test_loader = get_dataset(batch_size, cuda,crop=crop,normalization=normalization)

    # TensorBoard writer
    writer_filename = (
            f"bs{batch_size}_lr{lr}_drop{dropout_rate}_norm{int(normalization)}_"
            f"crop{int(crop)}_decay{lr_decay}_bn{int(batch_norm)}"
        )
    writer = SummaryWriter(log_dir=f"runs_new/{writer_filename}")

    for epoch_idx in range(epochs):
        print(f"\n===== EPOCH {epoch_idx + 1}/{epochs} =====")
        save_filename = (
            f"epoch{epoch_idx}_bs{batch_size}_lr{lr}_drop{dropout_rate}_norm{int(normalization)}_"
            f"crop{int(crop)}_decay{lr_decay}_bn{int(batch_norm)}"
        )


        loss_plot = TrainLossPlot(save_file_name=save_filename)


        df_data['batch_size'].append(batch_size)
        df_data['lr'].append(lr)
        df_data['dropout_rate'].append(dropout_rate)
        df_data['normalization'].append(normalization)
        df_data['crop'].append(crop)
        df_data['lr_decay'].append(lr_decay)
        df_data['batch_norm'].append(batch_norm)
    
        # Train phase
        train_top1, train_top5, train_loss = epoch(train_loader, model, criterion, optimizer, cuda,epoch_idx)

        # Eval phase
        test_top1, test_top5, test_loss = epoch(test_loader, model, criterion, cuda=cuda,epoch_idx=epoch_idx)

        # Logowanie do TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch_idx)
        writer.add_scalar('Loss/test', test_loss, epoch_idx)
        writer.add_scalar('Accuracy/train_top1', train_top1, epoch_idx)
        writer.add_scalar('Accuracy/test_top1', test_top1, epoch_idx)

        df_data['trainloss'].append(train_loss)
        df_data['testloss'].append(test_loss)
        df_data['train_acc'].append(train_top1)
        df_data['test_top1'].append(test_top1)
        df_data['epoch'].append(epoch_idx)

        lr_scheduler.step()
        loss_plot.plot()
        loss_plot.save()

    writer.close()



global df_data
df_data = defaultdict(list)

if __name__ == "__main__":
    """
    # varying batch size and learning rate 
    for batch_size in [32,64,128,256]:
        for lr in [1e-1,1e-2]:
            main(batch_size=batch_size, lr=lr, epochs=20)

    # normalization 
    main(batch_size=128,lr=0.1,epochs=20,normalization=True)


    # data augmentation 
    main(batch_size=128,lr=0.1,epochs=20,normalization=True, crop=True)

    # lr scheduler
    for lr_decay in [0.8,0.9,0.95]:
        main(batch_size=128,lr=0.1,epochs=20,normalization=True, crop=True,lr_decay=lr_decay)

    # dropout

    for dropout in [0.3,0.4,0.5]:
        main(batch_size=128,lr=0.1,epochs=20,normalization=True,crop=True,lr_decay=0.95,dropout_rate=dropout)

    # batch norm
    """
    main(batch_size=128,lr=0.1,epochs=20,normalization=True,crop=True,lr_decay=0.95,dropout_rate=0.5,batch_norm=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame(data=df_data)
    df.to_pickle(f"runs_new/{timestamp}.pckl")
    

