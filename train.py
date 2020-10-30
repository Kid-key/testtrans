import argparse
import os
import shutil
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datetime import datetime
TIME_NOW = datetime.now().isoformat()
#if not os.path.exists(TIME_NOW):
#    os.mkdir(TIME_NOW)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

#os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
best_prec1 = 0
args = parser.parse_args()
import numpy as np
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def getdataset():
    
    from torch.utils.data import Dataset
    class Subset(Dataset):
        """
        Subset of a dataset at specified indices.

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
        """
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    #from torch.utils.data.sampler import SubsetRandomSampler
    import torchvision.transforms as transforms
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    #traindir = os.path.join('/lustre/home/acct-seedwr/seedwr/imagenetdata','%s.lmdb'%'train')
    #valdir = os.path.join('/lustre/home/acct-seedwr/seedwr/imagenetdata','%s.lmdb'%'val')
    #valdir = '/home/data/val'
    imf=datasets.ImageFolder
    #imf=ImageFolderLMDB
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    train_dataset = imf(traindir,
                transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True) 


    val_loader = torch.utils.data.DataLoader(
       imf(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=args.batch_size, shuffle=False,num_workers=8, pin_memory=True)
    return train_loader,val_loader


from resnet import resnet18,resnet34
import quant_utils
if '18' in args.arch:
    net = resnet18
else:
    net = resnet34

def main():
    global args, best_prec1
    #Bits=[10, 8, 7, 8, 6, 7, 5, 10, 6, 4, 5, 4, 5, 4, 3, 3, 3, 5, 3, 3]# 37527424.0/11166912
    #Bits=[10, 9, 8, 9, 8, 8, 7,  9, 7, 6, 6, 5, 7, 4, 4, 4, 3, 8, 3, 3]# 41627520.0
    Bits=[9,7,6,6,5,5,5, 7,5,9,4,4,4,4, 5,4,5,5,5, 4,4,4,4,4,4,4,4,4, 3,4,4,4,4,3,4,3]
    #Bits=4

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = net(pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = net()
    
    print(Bits)
    model,sf_list,n_dict=quant_utils.quant_model_bit(model,Bits)

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    train_loader,val_loader = getdataset()
    
   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            sf_list = checkpoint['sf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #quant_utils.quant_relu_module_bit(model, 4)
    quant_utils.quant_relu_module(model, n_dict)
    model.cuda()
    quant_utils.running_module(model)
    model.eval()
    validate(val_loader, model, criterion)
    quant_utils.test_module(model) 

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader,val_loader = getdataset()  

    if args.evaluate:
        model.eval()
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        consine_learning_rate(optimizer, epoch,args.lr,args.epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,sf_list,Bits)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)


def train(train_loader, model, criterion, optimizer, epoch,sf_list,Bits):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    best_prec1 = 0.

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = 1 # np.random.rand(1)
        if r < 0.5:
            lam = 0.8
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)


        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        quant_utils.changemodelbit_fast(Bits,model,sf_list)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input, target = input.to('cuda'), target.to('cuda')
            # compute output from model
            output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename=TIME_NOW+'/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, TIME_NOW+'/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import math
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(optimizer.param_groups[0]['lr'])
def linear_learning_rate(optimizer, epoch, init_lr,T_max=100):
    lr = 2e-5 - init_lr/T_max*(epoch+1-T_max)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def consine_learning_rate(optimizer, epoch, init_lr=0.1,T_max=120):
   """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
   lr = 1e-6 + init_lr*(1+math.cos(math.pi*epoch/T_max))/2
   for param_group in optimizer.param_groups:
       param_group['lr'] = lr
       print(optimizer.param_groups[0]['lr'])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print(output[:,1:10])
    #error("stop!")
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
