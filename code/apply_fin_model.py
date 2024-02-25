import numpy as np
import torch

from utils import *


"""
This script runs through the models end to end and returns the binary result
of the defect detection.
"""


source_dir = input("Enter file path of desired image: ") 
model_loc = input("Enter file path of windowed model weights ('.pt' or 'pth'): ") 
result_model_loc = input("Enter file path of result model weights ('.pt' or 'pth'): ") 


# transform image
filt_im = image_to_dwtchannels(source_dir)

# get networks from paths
net, result_net = get_networks(model_loc, result_model_loc)

# process windows through window model and return 84 channel activation result for image
channels_windowed = retrieve_window_info(net,filt_im)

result_class = result_net(channels_windowed).detach().numpy()
result_class = (result_class[:,1]>result_class[:,0])[0]

if result_class:
    print("Defect Detected")
else:
    print("No Defect Detected")