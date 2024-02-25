import numpy as np
import cv2
import pywt
import os


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

def process_dataset(source_folder, out_folder):
    # applies wavelet transform at level 2 with db3 wavelet base to each image.
    # saves results as multichannel image with appropriate resizing in out_folder.

    defect_images_folder = os.path.join(source_folder, "Defect_images")

    for file_name in os.listdir(defect_images_folder):
        source_path = os.path.join(defect_images_folder, file_name)
        dest_path = os.path.join(out_folder, file_name[:-3]+"npy")
        img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

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
        np.save(dest_path,out_arr)

    # apply same processes to defect-less images
    no_defect_images_folder = os.path.join(source_folder, "NODefect_images")
    
    for subfolder_name in os.listdir(no_defect_images_folder):
        subfolder_path = os.path.join(no_defect_images_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                source_path = os.path.join(subfolder_path, file_name)
                dest_path = os.path.join(out_folder, file_name[:-3]+"npy")
                img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
                blur_img = cv2.GaussianBlur(img,(11,11),0)
                wavelets = pywt.wavedec2(blur_img, 'db3',  level=2)
                out_arr = np.zeros((7,40,500))
                out_arr[0,:,:] =cv2.resize(wavelets[0], (500,40))
                for i in range(3):
                    out_arr[i+1,:,:] = cv2.resize(wavelets[1][i], (500,40))
                    out_arr[i+4,:,:] = cv2.resize(wavelets[2][i], (500,40))
                np.save(dest_path,out_arr)


def process_masks(source_folder,out_folder):
    # takes all masks in mask folder and resizes them to match the 
    # wavelet transform shape. creates blank masks for defect free samples.
    # saves results in out_folder.

    mask_images_folder = os.path.join(source_folder, "Mask_images")
    for file_name in os.listdir(mask_images_folder):
        source_path = os.path.join(mask_images_folder, file_name)
        dest_path = os.path.join(out_folder, file_name[:-3]+"npy")
        img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

        # resizing with interpolation in order to preserve masks as much
        # as possible upon resize
        img = cv2.resize(img,(2000,160),interpolation=cv2.INTER_LANCZOS4)
        img = cv2.resize(img, (500,40), interpolation=cv2.INTER_LANCZOS4)

        # all nonzero elements set to 1 in small mask
        img = np.where(img>0, 1, img)

        np.save(dest_path,img)
    
    # create blank masks for non-defect images 
    no_defect_images_folder = os.path.join(source_folder, "NODefect_images")
    for subfolder_name in os.listdir(no_defect_images_folder):
        subfolder_path = os.path.join(no_defect_images_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                dest_path = os.path.join(out_folder, file_name[:-4]+"_mask.npy")
                np.save(dest_path, np.zeros((40,500)))

process_dataset(source_dir, dest_dir)
process_masks(source_dir, mask_dir)
