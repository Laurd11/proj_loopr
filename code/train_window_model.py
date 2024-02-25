import numpy as np
import torch
import os

from utils import *

source_dir = input("Enter directory with processed wavelet images: ") 
mask_dir  = input("Enter directory with processed mask images: ") 
dest_dir = input("Enter desired directory for model outputs: ") 
epchs = int(input("Enter desired number of epochs: "))

net = Net()
X_train, X_val, X_test, labels, val_labels, test_labels = get_processed_data(source_dir, mask_dir)
print("All data Loaded")

trained_net, losses, val_losses = training_procedure(net, X_train, X_val, labels, val_labels,dest_dir, total_epochs = epchs)
print('Finished Training')

# save all related info
torch.save(trained_net.state_dict(), os.path.join(dest_dir,"trained_net_"+str(epchs)+"_epo.pth"))
np.save(os.path.join(dest_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(dest_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

final_test(trained_net, X_test,test_labels)