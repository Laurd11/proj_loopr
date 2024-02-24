import numpy as np
import torch
import os

from utils import *

source_dir = input("Enter directory with processed wavelet images: ") 
model_loc = input("Enter file path of model weights ('.pt' or 'pth'): ") 
dest_dir = input("Enter desired directory for model outputs: ") 

img_list = retrieve_images(source_dir)
name_list = sorted(os.listdir(source_dir))

net = Net()
net.load_state_dict(torch.load(model_loc))
net.eval()

activations = []
def retrieveActivations():
    def hook(model, input, output):
        activations.append(output.detach().numpy())
    return hook

hook_layer = net.fc2.register_forward_hook(retrieveActivations())


for el in range(len(name_list)):
    im_win, _ = get_windowed_data(img_list[el,:,:,:].reshape(1,7,40,500),np.zeros(((1,1,40,500))))
    im_win = torch.from_numpy(im_win).type(torch.FloatTensor)
    outs_np = net(im_win)

hook_layer.remove()

for el in range(len(activations)):
    outs_np = activations[el].transpose().reshape((84,10,240))
    np.save(os.path.join(dest_dir,name_list[el]),outs_np)