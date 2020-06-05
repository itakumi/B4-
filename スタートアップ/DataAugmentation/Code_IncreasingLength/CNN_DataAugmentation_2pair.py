# -*- coding: utf-8 -*-
import sys, os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
n_kinds = 2
Data_Augmentation = "CenterCrop_RandomPerspective"
MODEL_PATH = "CNNmodel_" + Data_Augmentation + ".pth.tar"
MODEL_CHECKPOINT_PATH = "CP_CNNmodel_" + Data_Augmentation + ".pth.tar"
def load_cifar10(batch=100):
    num_workers = 0
    valid_size = 0.2
    #74％
    #train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomResizedCrop(32, scale=(0.08,1.0), ratio=(0.75,1.3)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    #ResizedCrop(倍率変換)
    #train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomResizedCrop(32, scale=(0.08,1.0), ratio=(0.75,1.3)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #CenterCrop
    train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.CenterCrop(random.randint(24,32)), transforms.RandomResizedCrop(32, scale=(1.0,1.0), ratio=(1.0,1.0)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomErasing(ランダムな範囲を塗りつぶす)
    train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomAffine(-30~30の間で回転)
    #train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomAffine([-30,30], scale=(0.8,1.2), shear=10), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomHorizontalFlip(確率pで左右反転)
    #train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomVerticalFlip
    #train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomPerspective(ランダムな視点から見た画像へと変換)
    #train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    train_data += datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.CenterCrop(random.randint(24,32)), transforms.RandomResizedCrop(32, scale=(1.0,1.0), ratio=(1.0,1.0)), transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    #train_data = train_data_normal + train_data_augmentation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    # trainとvalidの境目(split)を指定
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]

    # samplerの準備
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    # data loaderの準備
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch,
                                               sampler = train_sampler, num_workers = num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch,
                                              sampler = valid_sampler, num_workers = num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch,
                                             num_workers = num_workers)

    return {'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader}

def save_checkpoint(state, filename = MODEL_CHECKPOINT_PATH):
    torch.save(state, filename)

class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = torch.nn.Linear(4*4*64, 500)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def train():
    print(Data_Augmentation)
    print("will begin training")
    for ep in range(epoch):
        train_loss_total = 0
        train_acc_total = 0
        valid_loss_total = 0
        valid_acc_total = 0
        net.train()
        loss = None
        for i, (images, labels) in enumerate(loader['train_loader']):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            # 出力と結果が一致している個数を計算
            _,pred = torch.max(output,1)
            acc = np.squeeze(pred.eq(labels.data.view_as(pred)).sum())
            train_acc_total += acc
            loss.backward()
            optimizer.step()
            # training lossの計算
            train_loss_total += loss.item() * images.size(0)
            if i % 100 == 0:
                print('Training log: {} epoch ({} / 50000 train. data). Loss: {}, Acc: {}'.format(epochstart + ep + 1,
                                                                                         (i + 1) * 128,
                                                                                         loss.item(),
                                                                                         acc)
                      )

        torch.save(net.state_dict(), MODEL_PATH)
        train_loss = train_loss_total / len(loader['train_loader'].sampler)
        train_acc = train_acc_total.item() / len(loader['train_loader'].sampler)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        net.eval()
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader['valid_loader'])):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs,labels) # 損失を計算
                # 出力と結果が一致している個数を計算
                _,pred = torch.max(outputs,1)
                acc = np.squeeze(pred.eq(labels.data.view_as(pred)).sum())
                valid_acc_total += acc
                # validation lossの計算
                valid_loss_total += loss.item() * images.size(0)

        valid_loss = valid_loss_total / len(loader['valid_loader'].sampler)
        valid_acc = valid_acc_total.item() / len(loader['valid_loader'].sampler)
        #acc = float(correct / 50000)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        save_checkpoint({ # save parameters
            'epoch': epochstart + ep + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'valid_loss': history['valid_loss'],
            'valid_acc': history['valid_acc'],
            'test_loss': history['test_loss'],
            'test_acc': history['test_acc'],
        })

def test():
    #correct = 0
    test_loss_total = 0
    test_acc_total = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval() # ネットワークを推論モードへ
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader['test_loader'])):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs,labels) # 損失を計算
            # 出力と結果が一致している個数を計算
            _,pred = torch.max(outputs,1)
            test_acc_total += np.squeeze(pred.eq(labels.data.view_as(pred)).sum())
            total += labels.size(0)
            test_loss_total += loss.item()*images.size(0)
            #correct += (predicted == labels).sum()
            c = (pred == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

    #acc = float(correct / 10000)
    test_loss = test_loss_total / len(loader['test_loader'].sampler)
    test_acc = test_acc_total.item() / len(loader['test_loader'].sampler)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    save_checkpoint({ # save parameters
        'epoch': epochstart + epoch,
        'state_dict': net.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'valid_loss': history['valid_loss'],
        'valid_acc': history['valid_acc'],
        'test_loss': history['test_loss'],
        'test_acc': history['test_acc'],
    })

    print('Accuracy of the network on the 10000 test images: {:.2f} %'.format(
        100 * test_acc_total.item() / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def plot():
    # 結果をプロット
    plt.figure()
    plt.plot(range(1, epochstart+epoch+1), history['train_loss'], label='train_loss', color='red')
    plt.plot(range(1, epochstart+epoch+1), history['valid_loss'], label='val_loss', color='blue')
    plt.title('CNN Training Loss [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('img3/{}kind_CNN_cifar10_loss{}_1to{}.png'.format(n_kinds,Data_Augmentation,epochstart+epoch))
    plt.close()
    print("loss graph is saved")

    plt.figure()
    plt.plot(range(1, epochstart+epoch+1), history['train_acc'], label='train_acc', color='red')
    plt.plot(range(1, epochstart+epoch+1), history['valid_acc'], label='val_acc', color='blue')
    plt.title('CNN Accuracies [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('img3/{}kind_CNN_cifar10_acc{}_1to{}.png'.format(n_kinds,Data_Augmentation,epochstart+epoch))
    print("acc graph is saved")
    plt.close()

if __name__ == '__main__':
    start = time.time()
    epochstart = 0
    epoch = 50
    set_epoch = 50 * (2**n_kinds)
    loader = load_cifar10()
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR10のクラス

    use_cuda=torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("device=",device)
    net: MyCNN = MyCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # ロスの計算
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.00005)
    history = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'test_loss': [],
    'test_acc': []
    }
    while True:
        flag = os.path.exists(MODEL_PATH)
        flag_checkpoint = os.path.exists(MODEL_CHECKPOINT_PATH)
        if flag & flag_checkpoint: #前回の続きから学習
            print('loading parameters...')
            #net.load_state_dict(torch.load(MODEL_PATH))
            source = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
            net.load_state_dict(source)
            print('parameters loaded')
            print("loading history...")
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
            epochstart = checkpoint['epoch']
            history['train_loss'] = checkpoint['train_loss']
            history['train_acc'] = checkpoint['train_acc']
            history['valid_loss'] = checkpoint['valid_loss']
            history['valid_acc'] = checkpoint['valid_acc']
            history['test_loss'] = checkpoint['test_loss']
            history['test_acc'] = checkpoint['test_acc']
            print('history loaded')
        else:
            print("保存されたデータがないか、正しくありません")
            print("初期から学習します")

        print(epochstart)
        print("train_loss=",len(history['train_loss']))
        print("test_acc=",history['test_acc'])
        train()
        test()
        plot()
        if epoch==50:
            print("breakします")
            break
        if ((epochstart + 50 >= set_epoch) & (len(history['test_acc']) >= 2)):
            print("set_epocに達しています")
            if ((history['test_acc'][-1]*100) - (history['test_acc'][-2]*100)) < 0.3:
                print("差が0.3以下です。終了です")
                break
            else:
                print("まだ学習を続けます")
        else:
            print("set_epochに達していません。まだ学習を続けます")

    print("test_acc_result=",history['test_acc'])
    print('end of {}'.format(Data_Augmentation))
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
