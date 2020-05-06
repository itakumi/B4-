# -*- coding: utf-8 -*-
import sys, os
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

num_workers = 0
batch_size = 100
# 検証データ用の割合
valid_size = 0.2
# torch.FloatTensor型へ変換
transform = transforms.ToTensor()
# datasetsからデータを取得
train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

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
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)

# NNの定義
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 隠れ層 (512)
        hidden_1 = 512
        hidden_2 = 512
        hidden_3 = 512
        hidden_4 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28*28, hidden_1)
        nn.init.kaiming_normal_(self.fc1.weight)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        nn.init.kaiming_normal_(self.fc2.weight)
        # linear layer (n_hidden -> hidden_3)
        self.fc3 = nn.Linear(hidden_2,hidden_3)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(hidden_3,hidden_4)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.fc5 = nn.Linear(hidden_4,10)
        nn.init.kaiming_normal_(self.fc5.weight)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        # activation functionとしてrelu
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # activation functionとしてrelu
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        #x = self.fc3(x)
        # activation functionとしてrelu
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        return x

def train():
    print("will begin training")
    # minimum validation loss を最初は無限とする
    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        # それぞれの値を初期化
        train_loss_total = 0
        train_acc_total = 0
        valid_loss_total = 0
        valid_acc_total = 0

        ###################
        # train the model #
        ###################
        model.train() # ネットワークを学習モードへ
        for data,label in train_loader:
            optimizer.zero_grad() # 勾配結果を0にリセット
            data = data.view(-1,28 * 28)
            output = model(data) #順伝搬:入力dataをもとに出力outputを計算
            loss = criterion(output,label) #　損失を計算
            # training lossの計算
            train_loss_total += loss.item() * data.size(0)
            # 出力と結果が一致している個数を計算
            _,pred = torch.max(output,1)
            train_acc_total += np.squeeze(pred.eq(label.data.view_as(pred)).sum())
            loss.backward() #逆伝搬の計算
            optimizer.step() #重みの更新
            #print("loss=",loss.item())
            #print("data.size=",data.size(0))
        train_loss = train_loss_total / len(train_loader.sampler)
        train_acc = train_acc_total.item() / len(train_loader.sampler)
         ######################
        # validate the model #
        ######################

        model.eval()  # ネットワークを推論モードへ
        with torch.no_grad():
            for data,label in valid_loader:
                data = data.view(-1,28 * 28)
                output = model(data) #順伝搬:入力dataをもとに出力outputを計算
                loss = criterion(output,label) # 損失を計算
                # validation lossの計算
                valid_loss_total += loss.item() * data.size(0)
                # 出力と結果が一致している個数を計算
                _,pred = torch.max(output,1)
                valid_acc_total += np.squeeze(pred.eq(label.data.view_as(pred)).sum())

        # epoch毎のloss and accuracyを計算
        valid_loss = valid_loss_total / len(valid_loader.sampler)
        valid_acc = valid_acc_total.item() / len(valid_loader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            epoch+1,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc
            ))

        # validation lossが減ったら更新
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'MLPmodel.pt')
            valid_loss_min = valid_loss

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

def test():
    # テスト
    test_loss_total=0.0
    test_acc_total=0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval() # ネットワークを推論モードへ
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1,28 * 28)
            output = model(data) # 順伝搬:入力dataをもとに出力outputを計算
            loss = criterion(output, target) # 損失を計算
            # 予想と正解を比べる
            _, pred = torch.max(output, 1) # indices
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            test_acc_total += np.squeeze(pred.eq(target.data.view_as(pred)).sum())
            # test lossの計算
            test_loss_total += loss.item() * data.size(0)
            # 0~9までそれぞれの正答率を計算
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    # test lossの平均を計算
    test_loss = test_loss_total / len(test_loader.sampler)
    test_acc = test_acc_total.item() / len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def plot():
    plt.figure()
    plt.plot(range(n_epochs), history['train_loss'], label='train_loss', color='red')
    plt.plot(range(n_epochs), history['valid_loss'], label='val_loss', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('MLP Training Loss and validation loss')
    plt.legend()
    plt.savefig('MLP_Loss.png')

    plt.figure()
    plt.plot(range(n_epochs), history['train_acc'], label='train_acc', color='red')
    plt.plot(range(n_epochs), history['valid_acc'], label='val_acc', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('MLP Training accuracy and validation accuracy')
    plt.legend()
    plt.savefig('MLP_Accuracy.png')

if __name__ == '__main__':
    # 学習回数
    print("実行開始")
    n_epochs = 50
    # 学習結果の保存用
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    # initialize the NN
    flag = os.path.isfile('MLPmodel.pt')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    model = Net()#.to(device)
    # loss function を求める
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    if flag:
        print('loading parameters...')
        model.load_state_dict(torch.load("MLPmodel.pt"))
        print('parameters loaded')
    if n_epochs != 0:
        train()
    test()
    if flag == False: #最初からの学習しかプロットしないようにする
        plot()
