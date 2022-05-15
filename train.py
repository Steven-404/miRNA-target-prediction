import torch
import numpy as np
from torch import utils
from torch.utils.data.dataset import random_split
from functions import *
from model import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


model = BiLSTM()
model_name='RNN'
writer = SummaryWriter(log_dir='Result/log',comment=model_name)
train_file = "Data/train and test/data_set.csv"
batch_size = 32
epochs = 150
device = 'cpu'
train_set = TrainDataset(train_file)
train_size = len(train_set) * 0.8
test_size = len(train_set) * 0.2
train_set, test_set = torch.utils.data.random_split(train_set, [52341, 13085])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=13085, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(epochs):
    epoch_loss= 0
    model.train().to(device)
    for step, ((mirna, mrna), label) in enumerate(train_loader):
        mirna, mrna, label = mirna.to(device, dtype=torch.float), mrna.to(device, dtype=torch.float), label.to(
            device)

        outputs = model(mirna, mrna)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * outputs.size(0)
        #total += label.size(0)
        #correct += (torch.max(outputs, 1)[1] == label).sum().item()
    return epoch_loss/step

def test():
    correct = 0
    total = 0
    model.eval().to(device)
    with torch.no_grad():
        for ((mirna, mrna), label) in test_loader:
            mirna, mrna, label = mirna.to(device, dtype=torch.float), mrna.to(device, dtype=torch.float), label.to(
                device)

            outputs = model(mirna, mrna)
            _, predicts = torch.max(outputs.data, 1)
            #loss = criterion(outputs, label)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #epoch_loss += loss.item() * outputs.size(0)

            total += label.size(0)

            correct += (predicts == label).sum().item()
            #correct += (torch.max(outputs, 1)[1] == label).sum().item()

        # print('Acc %.3f %% [%d/%d]' % (100 * correct / total,correct,total))
        return correct/total,label.data,predicts.data

if __name__ == '__main__':
    # a = torch.randn(32,30, 50)
    # b = torch.randn(32,30, 50)
    # c = (a, b)
    # writer.add_graph(model,c)
    for epoch in range(epochs):
        train_loss= train(epoch)
        print('train_loss:',train_loss)
        #print('train_acc:',train_acc)
        writer.add_scalar("train_loss", train_loss, epoch)
        #writer.add_scalar("train/acc", train_acc, epoch)
        test_acc,test_label,test_predict = test()

        #print('test_loss:',test_loss)
        print('test_acc:',test_acc)
        #writer.add_scalar("test/loss", test_loss ,epoch)
        writer.add_scalar("test_acc", test_acc, epoch)
        writer.add_pr_curve('confusion matrix',test_label,test_predict,epoch)
