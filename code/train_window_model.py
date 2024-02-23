import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


epchs = 50

def retrieve_images(fil_dir):
    sample_list = os.listdir(fil_dir)
    img_list = np.zeros((len(sample_list),7,40,500))
    ind = 0
    for file_name in sample_list:
        if file_name[5:8] != "000":
            img_list[ind,:,:,:] = np.load(os.path.join(fil_dir,file_name))
            ind+=1
    return img_list[:ind,:,:,:]

def retrieve_masks(fil_dir):
    sample_list = os.listdir(fil_dir)
    img_list = np.zeros((len(sample_list),40,500))
    ind = 0
    for file_name in sample_list:
        if file_name[5:8] != "000":
            img_list[ind,:,:] = np.load(os.path.join(fil_dir,file_name))
            ind+=1
    return img_list.reshape(-1,1,40,500)[:ind,:,:,:]

def sliding_window(image, sq_size = 22, pix_steps=2):
    img_list = []
    for i in np.arange(0,image.shape[1]-sq_size+pix_steps,pix_steps):
        for j in np.arange(0,image.shape[2]-sq_size+pix_steps,pix_steps):
            img_list.append(image[:,i:i+sq_size,j:j+sq_size])
    return np.array(img_list)


def get_windowed_data(imgs, masks):
    windows_per_im = 10*240
    fin_data = np.zeros((imgs.shape[0]*windows_per_im,7,22,22))
    fin_labs = np.zeros((imgs.shape[0]*windows_per_im,1))
    for ind in range(imgs.shape[0]):
        fin_data[ind*windows_per_im:(ind+1)*windows_per_im,:,:,:] = sliding_window(imgs[ind,:,:,:])
        mask_windows = sliding_window(masks[ind,:,:,:])
        for el in range(mask_windows.shape[0]):
            if np.amax(mask_windows[el])>0:
                fin_labs[ind*windows_per_im+el]=1
    return fin_data, fin_labs
        
class Net(nn.Module):
    def __init__(self):
        super().__init__() # 7 x 22 x 22
        self.conv1 = nn.Conv2d(7, 16, 4,stride=2) # 16 x 10 x 10
        self.pool = nn.MaxPool2d(2,stride=1) # 16 x 9 x 9
        self.conv2 = nn.Conv2d(16, 32, 3,stride=2) # 32 x 4 x 4
        self.drop = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear( 512, 120)   
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1) 
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def get_processed_data(source_dir, mask_dir, test_split = 0.2,val_split=0.15):
    xs = retrieve_images(source_dir)
    masks = retrieve_masks(mask_dir)

    print("Images and Masks Retrieved")

    X_train, X_test, y_train, y_test = train_test_split(xs, masks, test_size=test_split, random_state=81)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=81)
    X_train, y_train = get_windowed_data(X_train, y_train)

    smote = SMOTE(random_state=81)
    X_train, y_train = smote.fit_resample(X_train.reshape(-1, 22*22*7), y_train)

    X_train = X_train.reshape(-1, 7, 22, 22)

    print("Data balanced")

    X_val, y_val = get_windowed_data(X_val, y_val)
    X_test, y_test = get_windowed_data(X_test, y_test)

    labels = np.zeros((y_train.shape[0],2))
    labels[:,1] = y_train.flatten()
    labels[np.argwhere(y_train==0).flatten(),0] = 1

    val_labels = np.zeros((y_val.shape[0],2))
    val_labels[:,1] = y_val.flatten()
    val_labels[np.argwhere(y_val==0).flatten(),0] = 1

    test_labels = np.zeros((y_test.shape[0],2))
    test_labels[:,1] = y_test.flatten()
    test_labels[np.argwhere(y_test==0).flatten(),0] = 1

    return X_train, X_val, X_test, labels, val_labels, test_labels

def training_procedure(net, X_train, X_val, labels, val_labels, dest_fol, total_epochs = epchs, learning_rate=5e-5):
    criterion  = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.tensor(labels))
    train_loader = DataLoader(train_dataset,batch_size=50,sampler = RandomSampler(train_dataset, replacement=False,num_samples=80000)) 
    val_dataset = TensorDataset(torch.from_numpy(X_val).type(torch.FloatTensor), torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=50)#,sampler = RandomSampler(train_dataset, replacement=False, num_samples=30000))  
    
    print("Training Initiated")

    losses = np.zeros(total_epochs)
    val_losses = np.zeros(total_epochs)
    for epoch in range(total_epochs):  
        net.train()
        running_loss = 0.0

        for batch, (inputs, lab) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        net.eval()
        with torch.no_grad():
            for batch, (val_inputs, val_labs) in enumerate(val_loader):
                val_outputs = net(val_inputs)
                valloss = criterion(val_outputs, val_labs)
                val_losses[epoch] +=valloss.item()
        if epoch % 5==0:
            print("Epoch "+str(epoch+1)+": \n Training Loss is: "+str(running_loss))
            print(" Validation Loss is: "+str(val_losses[epoch] ))
            torch.save(net.state_dict(), os.path.join(dest_fol,"net_"+str(epoch+1)+"_epo.pth"))

        losses[epoch] = running_loss

    return net, losses, val_losses

def final_test(net, X_test,y_test):
    net.eval()
    test_dataset = TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=50)

    num_correct = 0
    true_neg = 0
    with torch.no_grad():
        for batch, (val_inputs, val_labs) in enumerate(test_loader):
            val_outputs = net(val_inputs).detach().numpy()
            val_outputs = (val_outputs[:,1]>val_outputs[:,0]).astype(int)
            num_correct += np.sum(val_labs.numpy()[:,1].flatten() * val_outputs.flatten())
            true_neg += np.sum(val_labs.numpy()[:,0].flatten() * (1-val_outputs.flatten()))
            # num_correct+=np.argwhere([:,1] ==val_labs.numpy()[:,1]).flatten().shape[0]
    acc = num_correct / np.sum(y_test[:,1])
    tn_rate = true_neg / np.sum(y_test[:,0])
    print("Sensitivity: "+str(acc))
    print("Specificity: "+str(tn_rate))


source_dir = input("Enter directory with processed wavelet images: ") 
mask_dir  = input("Enter directory with processed mask images: ") 
dest_dir = input("Enter desired directory for model outputs: ") 

net = Net()
X_train, X_val, X_test, labels, val_labels, test_labels = get_processed_data(source_dir, mask_dir)
print("All data Loaded")

trained_net, losses, val_losses = training_procedure(net, X_train, X_val, labels, val_labels,dest_dir)
print('Finished Training')

torch.save(trained_net.state_dict(), os.path.join(dest_dir,"trained_net_"+str(epchs)+"_epo.pth"))
np.save(os.path.join(dest_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(dest_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

final_test(net, X_test,test_labels)