import argparse
import os

import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            View((-1, 128 * 7 * 7)),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.net(x)

def set_seed(inc, base_seed=666666666):
    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)

def train(args):
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor())
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


    set_seed(0)
    model = Net()
    model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])
    model_without_ddp = model.module
    set_seed(args.global_rank)
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_without_ddp.parameters(), 1e-4)

    verbose = (args.global_rank == 0)  # print only on main process

    while True:
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()

            if verbose and i % 100 == 0:
                
                print('loss: {:.4f}, memory: {:.1f} MB'.format(loss_value, torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    args = parser.parse_args()

    print(args.batch_size, args.num_workers)

    args.global_rank = int(os.environ["RANK"])
    args.global_world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    
    torch.distributed.init_process_group(backend='nccl')
    torch.distributed.barrier()

    torch.cuda.set_device(args.local_rank)
    print('DDP worker initialized as global_rank {}/{}, local_rank {}/{}'.format(args.global_rank, args.global_world_size, args.local_rank, args.local_world_size), flush=True)

    train(args)
