import numpy as np
import cv2
import pywt
import os

from utils import *
from pre_processing import process_dataset, process_masks

"""
This script executes the training process from end to end, starting with the images in the archive.
"""

source_dir = input("Enter archive location in disk: ") 
dest_dir  = input("Enter wavelet image output location in disk: ") 
mask_dir  = input("Enter mask output location in disk: ") 
window_model_dir = input("Enter desired directory for window model outputs: ") 
window_out_loc = input("Enter desired directory for resulting model image data: ") 
result_model_dir = input("Enter desired directory for result model outputs: ") 


# PRE PROCESSING

process_dataset(source_dir, dest_dir)
process_masks(source_dir, mask_dir)

print("Processed DWT All Images")

# WINDOW MODEL TRAINING

epchs = 70

# create CNN instance
net = Net()

# process data into windows and retrieve different subsets for training
X_train, X_val, X_test, labels, val_labels, test_labels = get_processed_data(dest_dir, mask_dir)
print("Windowed data Loaded")

# training
trained_window_net, losses, val_losses = training_procedure(net, X_train, X_val, labels, val_labels,window_model_dir, total_epochs = epchs)
print('Finished Windowed Model Training')

# save all related info
window_model_pth= os.path.join(window_model_dir,"trained_net_"+str(epchs)+"_epo.pth")
torch.save(trained_window_net.state_dict(), window_model_pth)
np.save(os.path.join(window_model_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(window_model_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

# print accuracy on test set
final_test(trained_window_net, X_test,test_labels)


# PROCESS WINDOWS THROUGH MODEL

# collect all the images and their names
img_list = retrieve_images(dest_dir)
name_list = sorted(os.listdir(dest_dir))

# load model with desired trained weights
trained_window_net.eval()

# set forward hook
activations = []
def retrieveActivations():
    def hook(model, input, output):
        activations.append(output.detach().numpy())
    return hook

hook_layer = trained_window_net.fc2.register_forward_hook(retrieveActivations())

# run through all images, apply sliding window, and store the window activations using the hook
for el in range(len(name_list)):
    im_win, _ = get_windowed_data(img_list[el,:,:,:].reshape(1,7,40,500),np.zeros(((1,1,40,500))))
    im_win = torch.from_numpy(im_win).type(torch.FloatTensor)
    outs_np = trained_window_net(im_win)

hook_layer.remove()

# save the activations in an appropriately named numpy array on disk
# 84 channel vectors for each window (there are 10 x 240 per image) are stored
for el in range(len(activations)):
    outs_np = activations[el].transpose().reshape((84,10,240))
    np.save(os.path.join(window_out_loc,name_list[el]),outs_np)
print("All Images processed through Windowed Model")

# RESULT MODEL TRAINING
epchs = 50

# get pre-processed data with 84 channel windowed images resulting from the first model
# separates them into sub datasets necessary for training.
X_train, X_val, X_test, labels, val_labels, test_labels = get_windowed_output(source_dir)
print("Activation data loaded")

# create CNN instance
model_out = Net_result()

# training
trained_net, losses, val_losses = training_procedure(model_out, X_train, X_val, labels, val_labels, result_model_dir, 
                                                     total_epochs = epchs, learning_rate=5e-5, sgd_batch = None)

print('Finished Result Model Training')

# save all related info
torch.save(trained_net.state_dict(), os.path.join(result_model_dir,"trained_net_"+str(epchs)+"_epo.pth"))
np.save(os.path.join(result_model_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(result_model_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

# print accuracy on test set
final_test(trained_net, X_test,test_labels)