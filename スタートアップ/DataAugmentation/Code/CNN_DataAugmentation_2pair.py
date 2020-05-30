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

MODEL_PATH = "CNNmodel.pth.tar"
MODEL_SAVE_PATH = "2kinds_CNNmodel_RandomVerticalFlip_RandomPerspective.pth.tar"
Data_Augmentation1 = "RandomVerticalFlip"
Data_Augmentation2 = "RandomPerspective"

def load_cifar10(batch=100):
    num_workers = 0
    valid_size = 0.2
    #74％
    train_data_normal = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    #ResizedCrop(倍率変換)
    #train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomResizedCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #CenterCrop
    #train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomErasing(ランダムな範囲を塗りつぶす)
    #train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomAffine(-30~30の間で回転)
    #train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomAffine([-30,30], scale=(0.8,1.2), shear=10), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomHorizontalFlip(確率pで左右反転)
    #train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomVerticalFlip
    train_data_augmentation1 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #RandomPerspective(ランダムな視点から見た画像へと変換)
    train_data_augmentation2 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    train_data_augmentation3 = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([ transforms.RandomVerticalFlip(p=0.5), transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    train_data = train_data_normal + train_data_augmentation1 + train_data_augmentation2 + train_data_augmentation3
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
    #                                         num_workers = num_workers)

    return {'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader}

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
    print(Data_Augmentation1)
    print(Data_Augmentation2)
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
                print('Training log: {} epoch ({} / 200000 train. data). Loss: {}, Acc: {}'.format(ep + 1,
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
    plt.title('CNN Training Loss [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('img/CNN_2kinds/2kinds_CNN_cifar10_loss{}_{}.png'.format(Data_Augmentation1,Data_Augmentation2))

    plt.figure()
    plt.plot(range(1, epoch+1), history['train_acc'], label='train_acc', color='red')
    plt.plot(range(1, epoch+1), history['valid_acc'], label='val_acc', color='blue')
    plt.title('CNN Accuracies [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('img/CNN_2kinds/2kinds_CNN_cifar10_acc{}_{}.png'.format(Data_Augmentation1,Data_Augmentation2))
    plt.close()

if __name__ == '__main__':
    start = time.time()
    epoch = 50
    loader = load_cifar10()
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR10のクラス
    """
    for i, (images, labels) in enumerate(tqdm(loader['train_loader_normal'])):
        print(images.shape)  # torch.Size([128, 3, 32, 32])

        # 試しに50枚を 5x10 で見てみる
        for i in range(5):
            for j in range(10):
                image = images[i*10+j] / 2 + 0.5
                image = image.numpy()
                plt.subplot(5, 10, i*10+j + 1)
                plt.imshow(np.transpose(image, (1, 2, 0)))  # matplotlibではチャネルは第3次元

                # 対応するラベル
                plt.title(classes[int(labels[i*10+j])])

                # 軸目盛や値はいらないので消す
                plt.tick_params(labelbottom=False,
                                labelleft=False,
                                labelright=False,
                                labeltop=False,
                                bottom=False,
                                left=False,
                                right=False,
                                top=False)
        plt.savefig('img/cifar10_normal{}.png'.format(Data_Augmentation))
        break

    for i, (images, labels) in enumerate(tqdm(loader['train_loader_augmentation'])):
        print(images.shape)  # torch.Size([128, 3, 32, 32])

        # 試しに50枚を 5x10 で見てみる
        for i in range(5):
            for j in range(10):
                image = images[i*10+j] / 2 + 0.5
                image = image.numpy()
                plt.subplot(5, 10, i*10+j + 1)
                plt.imshow(np.transpose(image, (1, 2, 0)))  # matplotlibではチャネルは第3次元

                # 対応するラベル
                plt.title(classes[int(labels[i*10+j])])

                # 軸目盛や値はいらないので消す
                plt.tick_params(labelbottom=False,
                                labelleft=False,
                                labelright=False,
                                labeltop=False,
                                bottom=False,
                                left=False,
                                right=False,
                                top=False)
        plt.savefig('img/cifar10_augmentation{}.png'.format(Data_Augmentation))
        break
    """

    use_cuda=torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("device=",device)
    net: MyCNN = MyCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # ロスの計算
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.00005)
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
    print('end of {}'.format(Data_Augmentation1))
    print('end of {}'.format(Data_Augmentation2))
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
