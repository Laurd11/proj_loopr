import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

source_dir = input("Enter processed data location in disk: ") 

def retrieve_classes(fil_dir):
    sample_list = os.listdir(fil_dir)
    class_list = np.zeros(len(sample_list))
    ind = 0
    for file_name in sample_list:
        if file_name[5:8] != "000":
            class_list[ind] = 1 #int(file_name[5:8])
        ind+=1
    return class_list

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 4, 2)
        self.pool = nn.MaxPool2d(2,stride=1)
        self.conv2 = nn.Conv2d(4, 10, 4,2)
        self.fc1 = nn.Linear( 1960, 120)   
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

net = Net()

criterion  = nn.CrossEntropyLoss() #DiceLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-5)

train_dataset = TensorDataset(torch.from_numpy(X_train.reshape(-1,1,32,32)), torch.tensor(labels))
train_loader = DataLoader(train_dataset,batch_size=50,sampler = RandomSampler(train_dataset, replacement=False, num_samples=50000)) 
val_dataset = TensorDataset(torch.from_numpy(X_test.reshape(-1,1,32,32)), torch.tensor(test_labels))
val_loader = DataLoader(val_dataset, batch_size=50) 

total_epochs = 50
losses = np.zeros(total_epochs)
for epoch in range(total_epochs):  

    running_loss = 0.0
    
    for batch, (inputs, lab) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    
    with torch.no_grad():
        val_loss = 0.
        for batch, (val_inputs, val_labels) in enumerate(val_loader):#val_sample in test_loader:
        #         #val_features, val_labels = next(iter(test_loader))
            val_outputs = net(val_inputs)
            valloss = criterion(val_outputs, val_labels)
            val_loss +=valloss.item()

    print("Epoch "+str(epoch+1)+": \n Training Loss is: "+str(running_loss))
    print(" Validation Loss is: "+str(val_loss))
    losses[epoch] = running_loss
print('Finished Training')