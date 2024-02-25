import numpy as np
import torch
import os

from utils import *

"""
This script trains a model that performs binary classification on the overall
images based on the processed data from the pre-trained windowed model. This
training is conducted on the whole dataset rather than just the defect images. 
"""


source_dir = input("Enter directory with outputs of windowed model: ") 
dest_dir = input("Enter desired directory for model outputs: ") 
epchs = int(input("Enter desired number of epochs: "))

# get pre-processed data with 84 channel windowed images resulting from the first model
# separates them into sub datasets necessary for training.
X_train, X_val, X_test, labels, val_labels, test_labels = get_windowed_output(source_dir)
print("Data loaded")

# create CNN instance
model_out = Net_result()

# training
trained_net, losses, val_losses = training_procedure(model_out, X_train, X_val, labels, val_labels, dest_dir, total_epochs = epchs, 
                                                     learning_rate=5e-5, sgd_batch = None)

print('Finished Training')

# save all related info
torch.save(trained_net.state_dict(), os.path.join(dest_dir,"trained_net_"+str(epchs)+"_epo.pth"))
np.save(os.path.join(dest_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(dest_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

# print accuracy on test set
final_test(trained_net, X_test,test_labels)