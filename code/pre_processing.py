from utils import *


"""
This file takes the archive in the format given on Kaggle and processes the
images using pywt.wavedec2 with a db3 wavelet filter to level 2. This results 
in 7 resulting filtered images that are then resized in order to form a 
(7 channel x 40 height x 500 width) image result for each image in the dataset.
The corresponding masks for each image are also resized, with interpolation and
thresholding where necessary. All results are saved in a single folder.

LOOPING CODE ADAPTED FROM NOTEBOOK: https://www.kaggle.com/code/nexuswho/aitex-defect-detection
"""


source_dir = input("Enter archive location in disk: ") 
dest_dir  = input("Enter image output location in disk: ") 
mask_dir  = input("Enter mask output location in disk: ") 

process_dataset(source_dir, dest_dir)
process_masks(source_dir, mask_dir)
