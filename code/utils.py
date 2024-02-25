import numpy as np
import torch
from torch import nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import cv2
import pywt

"""
This file contains functions and classes utilized in data processing and model training
"""


# BASIC PROCESSED IMAGE LOADING FUNCTIONS


def retrieve_images(fil_dir, exclude_clean=False, shape=(7,40,500)):
    # gets processed multichannel images from a single directory
    # flexible shape, exclude_clean is used to get defect images only
    # returns numpy array (# images x #channels x height x width)

    sample_list = sorted(os.listdir(fil_dir))
    img_list = np.zeros((len(sample_list),shape[0],shape[1],shape[2]))
    ind = 0
    if exclude_clean:
        # only retrieves images with defects
        for file_name in sample_list:
            if file_name[5:8] != "000":
                img_list[ind,:,:,:] = np.load(os.path.join(fil_dir,file_name))
                ind+=1
    else:
        # retrieves all images in directory
        for file_name in sample_list:
            img_list[ind,:,:,:] = np.load(os.path.join(fil_dir,file_name))
            ind+=1
    return img_list[:ind,:,:,:]

def retrieve_masks(fil_dir, exclude_clean=False):
    # gets processed masks from a directory, same exclude_clean option as above
    # shape is specific to single channel masks corresponding to wavelet filters
    # returns numpy array (# images x 1 x 40 x 500)

    sample_list = sorted(os.listdir(fil_dir))
    img_list = np.zeros((len(sample_list),40,500))
    ind = 0
    if exclude_clean:
        for file_name in sample_list:
            if file_name[5:8] != "000":
                img_list[ind,:,:] = np.load(os.path.join(fil_dir,file_name))
                ind+=1
    else:
        for file_name in sample_list:
            img_list[ind,:,:] = np.load(os.path.join(fil_dir,file_name))
            ind+=1
    return img_list.reshape(-1,1,40,500)[:ind,:,:,:]


# DATA METHODS AND MODEL FOR SLIDING WINDOW MODEL


def sliding_window(image, sq_size = 22, pix_steps=2):
    # helper function that takes a multichannel numpy array image and goes left to
    # right row by row extracting windowed subimages of size sq_size
    # pix steps is the number of steps taken between each column and each row of windows
    # when pix_steps<sq_size, the windows overlap
    # Returns: numpy array of size (#total windows x sq_size x sq_size)

    img_list = []
    for i in np.arange(0,image.shape[1]-sq_size+pix_steps,pix_steps):
        for j in np.arange(0,image.shape[2]-sq_size+pix_steps,pix_steps):
            img_list.append(image[:,i:i+sq_size,j:j+sq_size])
    return np.array(img_list)


def get_windowed_data(imgs, masks):
    # helper function that takes arrays imgs (#images x #chnnels x height x width)
    # and masks  (#images x 1 x height x width) and applies sliding_window to both.
    # The sliding windows of imgs are returned (#windows x #channels x 22 x 22)
    # the labels of each window are returned (#windows x 1)

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
    # CNN applied to 7x22x22 windows of wavelet filtered data,
    # outputs a binary classification. second to last activation used for 
    # final image classification.
    
    def __init__(self):
        # output sizes listed to the right
        super().__init__() # 7 x 22 x 22
        self.conv1 = nn.Conv2d(7, 16, 4,stride=2) # 16 x 10 x 10
        self.pool = nn.MaxPool2d(2,stride=1) # 16 x 9 x 9
        self.conv2 = nn.Conv2d(16, 32, 3,stride=2) # 32 x 4 x 4
        # flatten vector
        self.drop = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear( 512, 120)   
        self.fc2 = nn.Linear(120, 84)
        # OUTPUT OF LAYER fc2 USED FOR RESULT MODEL
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
    # this function takes directories containing all processed images (the 7x40x500 wavelet images
    # and the 40x500 corresponding masks) and creates sliding window 7x22x22 images and binary 
    # classification of defect presence (1x2) for each window. The result is 10 * 240 sliding window
    # results per image. Since there are significantly more defect free images than defect ones, 
    # imblearn.over_sampling.SMOTE is used to balance the distribution

    # get images and masks from directories, ONLY USES IMAGES WITH DEFECTS via exclude_clean
    xs = retrieve_images(source_dir, exclude_clean=True)
    masks = retrieve_masks(mask_dir, exclude_clean=True)

    print("Images and Masks Retrieved")

    X_train, X_test, y_train, y_test = train_test_split(xs, masks, test_size=test_split, random_state=81)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=81)
    X_train, y_train = get_windowed_data(X_train, y_train)

    # SMOTE used to artificially balance positive/negative sample distribution
    smote = SMOTE(random_state=81)
    X_train, y_train = smote.fit_resample(X_train.reshape(-1, 22*22*7), y_train)

    X_train = X_train.reshape(-1, 7, 22, 22)

    print("Data balanced")

    # validation data windowing
    X_val, y_val = get_windowed_data(X_val, y_val)
    X_test, y_test = get_windowed_data(X_test, y_test)

    # reshapes the labels for all the windowed sub data sets
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


# TRAINING FUNCTIONS FOR BOTH MODELS


def training_procedure(net, X_train, X_val, labels, val_labels, dest_fol, total_epochs = 60, learning_rate=5e-5, sgd_batch = 80000):
    # basic training procedure for NN, training, and validation sets input. Expects numpy arrays for features and labels
    # Requires dest_fol file path for outputs of model save points 
    # returns trained network, training loss vector history, and validation loss vector history

    # Utilizes basic binary Cross Entropy Loss and Adam optimizer as well as fixed batch size of 50

    # An optional SGD_batch size is included for large training sets, such that a random fixed sample size of
    # the training set can be taken at each epoch in order to make best use of computational resources
    # If full training set desired, set sgd_batch to None

    criterion  = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.tensor(labels))

    # this data loader uses RandomSampler to randomize the ordering of the input data for each epoch
    train_loader = DataLoader(train_dataset,batch_size=50,sampler = RandomSampler(train_dataset, 
                                                                                  replacement=False,num_samples=sgd_batch)) 
    
    val_dataset = TensorDataset(torch.from_numpy(X_val).type(torch.FloatTensor), torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=50)  
    
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

        # validation loss 
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
    # basic binary classification accuracy assessment, prints results from a numpy
    # array test set given for the NN provided. Assumes y_test is (#samples x 2)
    # with column 0 as negative results and column 1 as positives

    net.eval()
    test_dataset = TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=50)

    num_correct = 0
    true_neg = 0
    with torch.no_grad():
        for batch, (val_inputs, val_labs) in enumerate(test_loader):
            val_outputs = net(val_inputs).detach().numpy()
            val_outputs = (val_outputs[:,1]>val_outputs[:,0]).astype(int)

            # number of true positives in batch
            num_correct += np.sum(val_labs.numpy()[:,1].flatten() * val_outputs.flatten())

            # number of true negatives in batch
            true_neg += np.sum(val_labs.numpy()[:,0].flatten() * (1-val_outputs.flatten()))

    acc = num_correct / np.sum(y_test[:,1])
    tn_rate = true_neg / np.sum(y_test[:,0])
    full_acc = (num_correct+true_neg) / y_test.shape[0]

    print("Sensitivity: "+str(acc))
    print("Specificity: "+str(tn_rate))
    print("Full Accuracy: "+str(full_acc))



# RESULT MODEL TRAINING


class Net_result(nn.Module):
    # CNN that takes activation output from the windowed model (84channels x 10height 
    # x 240width) and produces a binary classification result (2 element probabilities) 
    # for the whole image

    def __init__(self):
        # output sizes listed to the right
        super().__init__() # input: 84 x 10 x 240
        self.conv1 = nn.Conv2d(84, 32, (1,2),stride=(1,2)) # 32 x 10 x 120 
        self.pool = nn.MaxPool2d(2) # 32 x 5 x 60
        self.conv2 = nn.Conv2d(32, 16, 2,stride=(1,2)) # 16 x 4 x 30 
        # flatten vector
        self.drop = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear( 1920, 360)   
        self.fc2 = nn.Linear(360, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x,1) 
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
def get_windowed_output(source_dir, test_split = 0.2, val_split=0.15):
    # function that fetches outputs of the initial windowed model from a path
    # source_dir and outputs training, validation, and test sets from this data
    # optional percentages 

    # xs is #images x 84 channels x 10height x 240width
    xs = retrieve_images(source_dir, shape=(84,10,240))
    name_list = sorted(os.listdir(source_dir))

    # this gets the image classifications from the file names
    ys = [1. if el[5:8]=="000" else 0. for el in name_list]
    ys = np.array(ys).reshape((-1,1))
    ys = np.hstack((ys, 1-ys))

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=test_split, random_state=81)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=81)

    return X_train, X_val, X_test, y_train, y_val, y_test


# SINGLE IMAGE PROCESSING THROUGH MODELS


def image_to_dwtchannels(img_loc):
    # converts single image to 7 channel discrete wavelet
    # transform array and returns it 

    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)

    # smooth image before applying DWT
    blur_img = cv2.GaussianBlur(img,(11,11),0)

    # apply discrete wavelet transform to image
    wavelets = pywt.wavedec2(blur_img, 'db3',  level=2)

    out_arr = np.zeros((7,40,500))

    # resize all wavelet transform outputs to smallest result shape
    out_arr[0,:,:] =cv2.resize(wavelets[0], (500,40))
    for i in range(3):
        out_arr[i+1,:,:] = cv2.resize(wavelets[1][i], (500,40))
        out_arr[i+4,:,:] = cv2.resize(wavelets[2][i], (500,40))

    return out_arr

def get_networks(model_dir, result_model_dir):
    # fetch two relevant networks

    net = Net()
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    result_net = Net_result()
    result_net.load_state_dict(torch.load(result_model_dir))
    result_net.eval()

    return net, result_net

def retrieve_window_info(net,img):
    # gets results of windowed model for single pre processed image

    net.eval()

    # set forward hook
    activations = []
    def retrieveActivations():
        def hook(model, input, output):
            activations.append(output.detach().numpy())
        return hook

    hook_layer = net.fc2.register_forward_hook(retrieveActivations())

    # apply sliding window to image and store the window activations using the hook
    im_win, _ = get_windowed_data(img.reshape(1,7,40,500),np.zeros(((1,1,40,500))))
    im_win = torch.from_numpy(im_win).type(torch.FloatTensor)
    outs_np = net(im_win)

    hook_layer.remove()

    # return the activations in a pytorch tensor with 1 element (84 x 10 x 240)
    
    outs_np = activations[0].transpose().reshape((84,10,240))
    outs_torch = torch.from_numpy(outs_np.reshape((1,84,10,240))).type(torch.FloatTensor)

    return outs_torch