# -*- coding: utf-8 -*-
import sys, os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
MODEL_PATH = "MLPmodel.pth.tar"
MODEL_SAVE_PATH = "MLPmodel_CenterCrop.pth.tar"
Data_Augmentation = "CenterCrop"

def load_cifar10(batch=100):
    num_workers = 0
    valid_size = 0.2
    #74％
    train_data_normal = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    #ResizedCrop(倍率変換)
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomResizedCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #CenterCrop
    train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.CenterCrop(random.randint(26,32)), transforms.RandomResizedCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomErasing(ランダムな範囲を塗りつぶす)
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomAffine(-30~30の間で回転)
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomAffine([-30,30], scale=(0.8,1.2), shear=10), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomHorizontalFlip(確率pで左右反転)
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomVerticalFlip
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomPerspective(ランダムな視点から見た画像へと変換)
    #train_data_augmentation = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    train_data = train_data_normal + train_data_augmentation
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

    #train_loader_normal = torch.utils.data.DataLoader(train_data_normal, batch_size = batch,
    #                                         num_workers = num_workers)
    #train_loader_augmentation = torch.utils.data.DataLoader(train_data_augmentation, batch_size = batch,
#                                             num_workers = num_workers)

    return {'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader}#, 'train_loader_normal': train_loader_normal, 'train_loader_augmentation': train_loader_augmentation}

class MyMLP(torch.nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        # 隠れ層 (512)
        hidden_1 = 1000
        hidden_2 = 1000
        hidden_3 = 1000
        hidden_4 = 200
        self.fc1 = torch.nn.Linear(16 * 3 * 8 * 8, hidden_1)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(hidden_1,hidden_2)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(hidden_2,hidden_3)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = torch.nn.Linear(hidden_3,hidden_4)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        self.fc5 = torch.nn.Linear(hidden_4,10)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        self.droput = torch.nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1,16 * 3 * 8 * 8)
        # activation functionとしてrelu
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
        # activation functionとしてrelu
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = F.relu(self.fc3(x))
        x = self.droput(x)
        x = F.relu(self.fc4(x))
        x = self.droput(x)
        x = self.fc5(x)
        return F.log_softmax(x,dim=1)

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

            if i % 10 == 0:
                print('Training log: {} epoch ({} / 100000 train. data). Loss: {}, Acc: {}'.format(ep + 1,
                                                                                         (i + 1) * 128,
                                                                                         loss.item(),
                                                                                         acc)
                      )

        torch.save(net.state_dict(), MODEL_SAVE_PATH)
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * test_acc_total.item() / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def plot():
    # 結果をプロット
    plt.figure()
    plt.plot(range(1, epoch+1), history['train_loss'], label='train_loss', color='red')
    plt.plot(range(1, epoch+1), history['valid_loss'], label='val_loss', color='blue')
    plt.title('MLP Training Loss [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('img/MLP_1kind/MLP_cifar10_loss{}.png'.format(Data_Augmentation))

    plt.figure()
    plt.plot(range(1, epoch+1), history['train_acc'], label='train_acc', color='red')
    plt.plot(range(1, epoch+1), history['valid_acc'], label='val_acc', color='blue')
    plt.title('MLP Accuracies [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('img/MLP_1kind/MLP_cifar10_acc{}.png'.format(Data_Augmentation))
    plt.close()

if __name__ == '__main__':
    start = time.time()
    epoch = 50
    loader = load_cifar10()
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR10のクラス

    use_cuda=torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("device=",device)
    net: MyMLP = MyMLP().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # ロスの計算
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.00005)
    flag = os.path.exists(MODEL_PATH)
    if flag: #前回の続きから学習
        print('loading parameters...')
        #net.load_state_dict(torch.load(MODEL_PATH))
        source = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        net.load_state_dict(source)
        print('parameters loaded')
    else:
        print("データがありません")
        print("初期から学習します")

    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    train()
    test()
    if flag == False:
        plot()
    print('end of {}'.format(Data_Augmentation))
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
