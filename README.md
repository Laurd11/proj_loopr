## Basic Model Overview:

* Each image is processed via Discrete Wavelet Transform into a 7 channel image that is 40 x 500, rather than 256 x 4096. Each channel represents a different level and directionality of wavelet.
* From these, sliding window squares (7 x 22 x 22) are extracted row and columnwise from  the preprocessed images. 
* The 7 channel squares from the Defect Images are used to train a "Windowed Model". The output is binary and the loss is computed based on the presence of any defects in that window given the information from the corresponding image mask.
* Once the Windowed Model is trained, we run it on all the images. The second to last activation outputs are extracted from this process, thus yielding an 84 element vector per window. 
* The result of extracting the Windowed Model activations from all the windows in an image is thus: 84 x 10 x 240
* This information is used to train another CNN on all the images, using their defect status as the label. This is called the Result Model.
* Applying the Windowed Model and the Result Model to an image yields a single Binary classification result. 

## Key Stats: 
* Sensitivity - 86.538%
* Specificity - 99.291%
* Total Accuracy - 93.878%

## Training Process and Final Model Information:
* The training process was completed using the files in order: 1) pre_processing.py, 2) train_window_model.py, 3) windowed_output.py, 4) train_result_model.py
* Most relevant functions and classes are stored in the utils.py file.
* One can conduct the whole process using the script end_to_end_training.py, although this does not yield the same flexibility of model selection in each step.
* Some data adjustments were committed prior to training, see Data Notes.
* The reported stats were yielded from the included weights in the repo. (Windowed Model - window_model_2_24_57tp_90tn/trained_net_70_epo.pth / Result Model - result_model_2_24_59tp_100tn/trained_net_50_epo.pth) They were applied to the whole dataset to retrieve the stats shown.

## Data Notes:
The following adjustments were made to the data:
* 0044_019_04 and 0097_030_03 had two masks associated, which were combined into a single mask for training.
* 0106_010_03 had no data in the mask and was removed from training
* 0100_025_08 had no associated mask and was removed from training
* 0018_00_01 was renamed to follow the appropriate naming convention prior to training

## Final Info:
apply_fin_model.py can be run to process a single image through the trained pipeline. The user selects the location of the trained network weights. Also, conda env info stored in environment.yml.
