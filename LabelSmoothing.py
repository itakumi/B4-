# -*- coding: utf-8 -*-
import sys, os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import warnings
import random
from tqdm import tqdm
import multiprocessing
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR,LambdaLR
import wandb
import pickle
NGPUS=int(os.environ['NGPUS'])
torch.manual_seed(0)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# afterphrase='LabelSmoothing'
# CHECKPOINT_PATH = "Checkpoint_"+afterphrase+".pth.tar"
# CHECKPOINT_SAVE_PATH = "Checkpoint_"+afterphrase+".pth.tar"
# MODEL_PATH = str(NGPUS)+"GPUResnet18_"+afterphrase+".pth.tar"
# MODEL_SAVE_PATH = str(NGPUS)+"GPUResnet18_"+afterphrase+".pth.tar"
# OPTIMIZER_PATH = str(NGPUS)+"GPUSGD_"+afterphrase+".pth.tar"
# OPTIMIZER_SAVE_PATH = str(NGPUS)+"GPUSGD_"+afterphrase+".pth.tar"
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--paper_k', type=int, default=NGPUS, metavar='M',
                    help='parameter k in the paper')
parser.add_argument('--paper_n', default=32, type=int, metavar='N',
                    help='parameter n in the paper')
parser.add_argument('--epsilon', default=0.1, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', type=float, default=1.0, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-b', '--batch-size', default=0, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--flooding', default=0.1, type=float, metavar='M',
                    help='flooding')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

history = {
    'lr':[],
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'test_loss': [],
    'test_acc': [],
}

best_acc1 = 0
# def linear_combination(x, y, epsilon):
#     return epsilon*x + (1-epsilon)*y
# def reduce_loss(loss, reduction='mean'):
#     return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = F.log_softmax(input, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)#onehot化
        if self.epsilon > 0.0:
            targets = ((1 - self.epsilon) * targets) + (self.epsilon / self.num_classes)
        targets = targets.detach()
        loss = (-targets * log_probs)

        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
def load_IMAGENET():
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val_ImageFolder')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    valid_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=valid_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize((224,224)),
    #         # transforms.Resize(256),
    #         # transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    if args.rank == 0:
        print(args)

    return {'train_loader': train_loader, 'valid_loader': val_loader}
def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(ddp_model,device,loader,criterion,optimizer,scheduler):
    if args.rank ==0:
        print("will begin training")
    flag_70=False
    flag_99=False
    for ep in range(len(history['train_loss']),len(history['train_loss'])+args.epochs):
    # for ep in range(args.epochs):
        start=time.time()
        train_loss_total = 0
        train_acc_total = 0
        valid_loss_total = 0
        valid_acc_total = 0
        # train_sampler.set_epoch(epoch)
        ddp_model.train()
        loss = None
        for i, (images, labels) in enumerate(loader['train_loader']):
            # train_sampler.set_epoch(epoch)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(args.gpu, non_blocking=True)
            optimizer.zero_grad()  # 勾配リセット
            outputs = ddp_model(images)  # 順伝播の計算
            loss = criterion(outputs, labels)  # lossの計算
            train_loss_total += loss*labels.size(0)  # train_loss に結果を蓄積
            acc = (outputs.max(1)[1] == labels).sum()  # 予測とラベルが合っている数の合計
            train_acc_total += acc  # train_acc に結果を蓄積
            loss.backward()  # 逆伝播の計算
            optimizer.step()  # 重みの更新
            if i % 10 == 0 and args.rank==0:
                print('Training log: {} epoch ({} / {} train. data). Loss: {}, Acc: {}'.format(ep + 1,
                                                                                         (i + 1) * args.batch_size,
                                                                                         len(loader['train_loader'])*args.batch_size,
                                                                                         loss.item(),
                                                                                         acc)
                      )
        elapsed_time = time.time() - start
        if args.rank==0:
            print ("learning time for 1 epoch =:{0}".format(elapsed_time) + "[sec]")
        dist.all_reduce(train_loss_total)
        dist.all_reduce(train_acc_total)
        train_loss_total=train_loss_total/NGPUS
        train_acc_total=train_acc_total//NGPUS
        # print(str(args.rank)+"のtrainlosstotalafter=",train_acc_total)
        # print(str(args.rank)+"のtrainacctotalafter=",train_acc_total)
        train_loss_total=train_loss_total.item()
        train_acc_total=train_acc_total.item()
        # if args.rank==0 and (ep+1)==args.epochs:
        #     print("最後にmodel,optimizerの保存")
        #     torch.save(ddp_model.state_dict(), MODEL_SAVE_PATH)
        #     torch.save(optimizer.state_dict(), OPTIMIZER_SAVE_PATH)
        train_loss = train_loss_total / len(loader['train_loader'].sampler)
        train_acc = train_acc_total / len(loader['train_loader'].sampler)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        start=time.time()
        ddp_model.eval()
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader['valid_loader']):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    labels = labels.cuda(args.gpu, non_blocking=True)
                # viewで28×28×１画像を１次元に変換し、deviceへ転送
                outputs = ddp_model(images) # 出力を計算(順伝搬)
                loss = criterion(outputs, labels) # lossを計算
                valid_loss_total += loss*labels.size(0) # lossを足す
                acc = (outputs.max(1)[1] == labels).sum() # 正解のものを足し合わせてaccを計算
                valid_acc_total += acc # accを足す
                if i % 10 == 0 and args.rank==0:
                    print('Valid log: {} epoch ({} / {} train. data). Loss: {}, Acc: {}'.format(ep + 1,
                                                                                             (i + 1) * args.batch_size,
                                                                                             len(loader['valid_loader'])*args.batch_size,
                                                                                             loss.item(),
                                                                                             acc)
                          )

        dist.all_reduce(valid_loss_total)
        dist.all_reduce(valid_acc_total)
        valid_loss_total=valid_loss_total/NGPUS
        valid_acc_total=valid_acc_total//NGPUS
        # print(str(args.rank)+"のvalidlosstotalafter=",valid_acc_total)
        # print(str(args.rank)+"のvalidacctotalafter=",valid_acc_total)
        valid_loss_total=valid_loss_total.item()
        valid_acc_total=valid_acc_total.item()
        valid_loss = valid_loss_total / len(loader['valid_loader'].sampler)
        valid_acc = valid_acc_total / len(loader['valid_loader'].sampler)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        elapsed_time = time.time() - start
        if args.rank==0:
            print ("valid time for 1 epoch =:{0}".format(elapsed_time) + "[sec]")
            wandb.log({
                    'epoch': ep+1,
                    'lr' : scheduler.get_last_lr()[0],
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'valid_loss': valid_loss,
                    'valid_accuracy': valid_acc
                    })
        scheduler.step()
        if args.rank == 0:
            print("valid_loss=",valid_loss)
            print("valid_acc=",valid_acc)
        # if valid_acc>=0.7 and flag_70==False:
        #     if args.rank == 0:
        #         print("70%over")
        #     flag_70=True
        #     torch.save(ddp_model.state_dict(), 'CNNmodel_checkpoint_70.pth.tar')
        # elif valid_acc>=0.99 and flag_99==False:
        #     if args.rank == 0:
        #         print("99%over")
        #     flag_99=True
        #     torch.save(ddp_model.state_dict(), "CNNmodel_checkpoint_99.pth.tar")
    if args.rank==0:
        save_checkpoint({
            'epoch': ep + 1,
            'arch': args.arch,
            'state_dict': ddp_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, CHECKPOINT_SAVE_PATH)

# def plot():
#     # 結果をプロット
#     plt.figure()
#     plt.plot(range(1, args.epochs+1), history['train_loss'], label='train_loss', color='red')
#     plt.plot(range(1, args.epochs+1), history['valid_loss'], label='val_loss', color='blue')
#     plt.title('Training Loss_'+afterphrase+' [IMAGENET]')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig("img/"+str(NGPUS)+'GPU_loss_'+afterphrase+'.png')
#
#     plt.figure()
#     plt.plot(range(1, args.epochs+1), history['train_acc'], label='train_acc', color='red')
#     plt.plot(range(1, args.epochs+1), history['valid_acc'], label='val_acc', color='blue')
#     plt.title('Accuracies'+afterphrase+' [IMAGENET]')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend()
#     plt.savefig("img/"+str(NGPUS)+'GPU_acc_'+afterphrase+'.png')
#     plt.close()
def setup(master_addr):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()  # number of processes (GPUs)
    rank = comm.Get_rank()  # ID of process
    init_method = 'tcp://{}:23456'.format(args.dist_url)

    # initialize the process group
    dist.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)

    return rank, world_size

def cleanup():
    dist.destroy_process_group()
def main():
    global args,best_acc1,history,afterphrase,CHECKPOINT_PATH,CHECKPOINT_SAVE_PATH
    # global args,best_acc1,history
    master_addr = sys.argv[1]
    args = parser.parse_args()
    afterphrase='LS'+str(args.epsilon)+str(args.arch)
    CHECKPOINT_PATH = "Checkpoint_"+afterphrase+".pth.tar"
    CHECKPOINT_SAVE_PATH = "Checkpoint_"+afterphrase+".pth.tar"

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if args.rank==0:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

    args.distributed = True
    if args.distributed:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        ngpus_per_node = torch.cuda.device_count()
        device = rank % ngpus_per_node
        torch.cuda.set_device(device)
        init_method = 'tcp://{}:23456'.format(args.dist_url)
        torch.distributed.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)
        args.rank = rank
        args.gpu = device
        args.batch_size=args.paper_n
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        args.world_size = world_size
    if args.rank==0:
        wandb.init(project="IMAGENET-Flood_PaperlikeProject", name=str(NGPUS)+"GPU_"+afterphrase) # 最初に呼び出して、プロジェクト名や実験名を指定する
        # wandb.config.update({"epochs": args.epochs,"batch_size": args.batch_size,"lr":args.lr,"gamma":args.gamma}) # 実験で使用したハイパラなどを入れる
        wandb.config.update(args) # 実験で使用したハイパラなどを入れる

    start = time.time()
    if args.rank == 0:
        print("実行開始")
    loader = load_IMAGENET()
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]().cuda(args.gpu)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model=model.to(device)
        # args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.workers = 4
        ddp_model = DDP(model, device_ids=[args.gpu]).cuda(args.gpu)
        ddp_model=ddp_model.to(device)
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion=LabelSmoothingCrossEntropy(num_classes=1000,epsilon=args.epsilon)
    # criterion=LabelSmoothingCrossEntropy(epsilon=args.epsilon).cuda(args.gpu)
    optimizer = torch.optim.SGD(params=ddp_model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    def func(epoch):
        if epoch==0:
            return 1
        if epoch <=5:
            if epoch==5:
                # args.lr*=args.paper_k*args.paper_n/256
                if args.rank==0:
                    print("return=",args.paper_k*args.paper_n/256)
                return args.paper_k*args.paper_n/256
            else:
                if args.rank==0:
                    print("return=",(1+(epoch*((args.paper_k*args.paper_n/256)-1)/5)))
                return (1+(epoch*((args.paper_k*args.paper_n/256)-1)/5))
                # return (args.lr+(args.lr*epoch*(args.paper_k-1)/5))
            # args.lr+=args.lr*(args.paper_k-1)/5
            # return args.lr
        elif epoch==30 or epoch==60 or epoch==80:
            args.gamma*=0.1
            return args.gamma*(args.paper_k*args.paper_n/256)
        else:
            return args.gamma*(args.paper_k*args.paper_n/256)
    scheduler = LambdaLR(optimizer, lr_lambda = func)
    flag = os.path.exists(CHECKPOINT_PATH)
    if flag: #前回の続きから学習
        if args.rank == 0:
            print('loading parameters...')
        #net.load_state_dict(torch.load(MODEL_PATH))
        checkpoint=torch.load(CHECKPOINT_PATH)
        # net.load_state_dict(torch.load(MODEL_PATH))
        # source = torch.load(MODEL_PATH)
        ddp_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # source = torch.load(MODEL_PATH)
        # ddp_model.load_state_dict(source)
        # source = torch.load(OPTIMIZER_PATH)
        # optimizer.load_state_dict(source)
        if args.rank == 0:
            print('parameters loaded')
            print("historyをロード")
        with open(str(NGPUS)+'GPU_history_'+afterphrase+'.pickle','rb') as f:
            history=pickle.load(f)
        if args.rank==0:
            for i in range(len(history['train_loss'])):
                wandb.log({
                        'epoch': i+1,
                        'lr':history['lr'][i],
                        'train_loss': history['train_loss'][i],
                        'train_accuracy': history['train_acc'][i],
                        'valid_loss': history['valid_loss'][i],
                        'valid_accuracy': history['valid_acc'][i],
                        # 'test_loss': test_loss,
                        # 'test_accuracy': test_acc
                        })
            # wandb.log({
            #         'test_loss_before': history['test_loss'][-1],
            #         'test_accuracy_before': history['test_acc'][-1]
            #         })
        for i in range(len(history['train_loss'])):
            scheduler.step()
    else:
        if args.rank == 0:
            print("途中のパラメータなし")
    cudnn.benchmark = True
    train(ddp_model,device,loader,criterion,optimizer,scheduler)
    # if flag == False:
    #     plot()
    elapsed_time = time.time() - start
    if args.rank==0:
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    if args.rank==0:
        print("historyを保存")
        with open(str(NGPUS)+'GPU_history_'+afterphrase+'.pickle','wb') as f:
            pickle.dump(history,f)
        cleanup()

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    main()
