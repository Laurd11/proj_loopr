import numpy as np
import torch
import os

from utils import *

"""
This script is run to bridge the training between the window model and the result image model.
It extracts the 84 channel activation outputs for each image from the first using a hook on 
the window model and then saves all the results in dest_dir.
"""

source_dir = input("Enter directory with processed wavelet images: ") 
model_loc = input("Enter file path of model weights ('.pt' or 'pth'): ") 
dest_dir = input("Enter desired directory for model outputs: ") 

# collect all the images and their names
img_list = retrieve_images(source_dir)
name_list = sorted(os.listdir(source_dir))

# load model with desired trained weights
net = Net()
net.load_state_dict(torch.load(model_loc))
net.eval()

# set forward hook
activations = []
def retrieveActivations():
    def hook(model, input, output):
        activations.append(output.detach().numpy())
    return hook

hook_layer = net.fc2.register_forward_hook(retrieveActivations())

# run through all images, apply sliding window, and store the window activations using the hook
for el in range(len(name_list)):
    im_win, _ = get_windowed_data(img_list[el,:,:,:].reshape(1,7,40,500),np.zeros(((1,1,40,500))))
    im_win = torch.from_numpy(im_win).type(torch.FloatTensor)
    outs_np = net(im_win)

hook_layer.remove()

# save the activations in an appropriately named numpy array on disk
# 84 channel vectors for each window (there are 10 x 240 per image) are stored
for el in range(len(activations)):
    outs_np = activations[el].transpose().reshape((84,10,240))
    np.save(os.path.join(dest_dir,name_list[el]),outs_np)