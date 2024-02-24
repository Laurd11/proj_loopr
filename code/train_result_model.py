import numpy as np
import torch
import os

from utils import *

source_dir = input("Enter directory with outputs of windowed model: ") 
dest_dir = input("Enter desired directory for model outputs: ") 
epchs = int(input("Enter desired number of epochs: "))

# fil_dir = "C:/Users/laure/Documents/proj_loopr/channeled_window_results"

X_train, X_val, X_test, labels, val_labels, test_labels = get_windowed_output(source_dir)
print("Data loaded")
model_out = Net_result()

trained_net, losses, val_losses = training_procedure(model_out, X_train, X_val, labels, val_labels, dest_dir, total_epochs = epchs, 
                                                     learning_rate=5e-5, sgd_batch = None)

print('Finished Training')

torch.save(trained_net.state_dict(), os.path.join(dest_dir,"trained_net_"+str(epchs)+"_epo.pth"))
np.save(os.path.join(dest_dir,"training_loss_"+str(epchs)+"_epo.npy"),losses)
np.save(os.path.join(dest_dir,"val_loss_"+str(epchs)+"_epo.npy"),val_losses)

final_test(trained_net, X_test,test_labels)