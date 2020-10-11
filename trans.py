from google.colab import drive
drive.mount('/gdrive', force_remount=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision 
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm as tqdm
import os
import pickle
import random

load = True
train_pick = None if load else train_pick
test_pick = None if load else test_pick 
if load:
    with open('/gdrive/My Drive/finale/TestTrainPick/train_pick.pkl', 'rb') as pickle_file:
        train_pick = pickle.load(pickle_file)
    with open('/gdrive/My Drive/finale/TestTrainPick/test_pick.pkl', 'rb') as pickle_file:
        test_pick = pickle.load(pickle_file)
    random.shuffle(train_pick)
    random.shuffle(test_pick)

class MakeData(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transforms = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        img = self.imgs[ind][0].copy()
        if self.transforms:
            img = self.transforms(img)
        return (img, self.imgs[ind][1])

def showTensorImage(tens):
    plt.imshow(tens.permute(1, 2, 0))

transformTrain = transforms.Compose([transforms.Resize((110, 110)), 
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
                                transforms.CenterCrop(105),
                                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.07),
                                transforms.RandomAffine(20, translate=(0.1,0.01)),
                                transforms.ToTensor()])
transformTest = transforms.Compose([transforms.Resize((110, 110)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(5),
                                    transforms.ToTensor()])

trainSet = MakeData(train_pick, transform=transformTrain)
testSet = MakeData(test_pick, transform=transformTest)

trainLoad = DataLoader(trainSet, shuffle=True, batch_size=8)
testLoad = DataLoader(testSet, batch_size=2000)

showTensorImage(trainSet[41][0])

"""Training the model"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, loader, epochs=5):
    ii = 0
    losses = []
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (1 / (1 + epoch*lrdec))
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
    return model, losses

# resnet34/50 lr=0.00001
lr = 0.000008
lrdec = 0.3

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 75)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)
model, losses = train_model(model, criterion, optimizer, trainLoad, epochs=10)

running = [losses[0]]
for i in range(1, len(losses)):
    running.append(0.99*running[i-1] + 0.01*losses[i])
plt.plot(losses, alpha=0.4, linewidth=0.4)
plt.plot(running, linewidth=3)

def get_accuracy(load, topx):
    total = 0
    correct = 0
    top5 = 0
    errors = []
    t5errors = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in tqdm(load):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).to('cpu')
            total += len(outputs)
            labels = labels.to('cpu')
            for i in range(len(outputs)):
                newOup = []
                for j in range(len(outputs[i])):
                    newOup.append([outputs[i][j], j])
                newOup.sort(key=lambda x : x[0], reverse=True)
                if newOup[0][1] == labels[i]:
                    correct += 1
                else:
                    errors.append((inputs[i], labels[i]))
                flag = False
                for t5 in range(topx):
                    if newOup[t5][1] == labels[i]:
                        top5 += 1
                        flag = True 
                if not flag:
                    t5errors.append((inputs[i], labels[i]))
    return (correct / total), (top5 / total), errors, t5errors

acc = get_accuracy(testLoad, 5)
acc2 = get_accuracy(trainLoad, 5)

print(acc[0], acc[1])
print(acc2[0], acc2[1])

if 1:
    showTensorImage(acc[3][16][0].to('cpu'))
    print(acc[3][16][1])

errors_inds = [0 for i in range(75)]
for img, label in acc[2]:
    errors_inds[label] += 1
print(sorted(errors_inds, reverse=True))

if False:
    torch.save(model.state_dict(), '/gdrive/My Drive/finale/Models/v2_t1_73_t5_93.pth')
    print("Saved")

!nvidia-smi