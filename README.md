# pictorial-maps-keras

## Installation

* Requires Python 3.6.x (https://www.python.org/downloads/)
* Requires CUDA Toolkit 9.0 (https://developer.nvidia.com/cuda-downloads) and corresponding cuDNN (https://developer.nvidia.com/rdp/cudnn-download)
* Download project
* pip install -r requirements.txt


## Inference

* Adjust models_folder in config.py
* Download trained models and place them inside the models folder

#### Maps vs. Non-maps
* Run classify_maps.py <input folder with images> <output folder for map and non-map images>

#### Pictorial maps vs. non-pictorial maps
* Run classify_pictorial_maps.py <input folder with map images> <output folder for pictorial map and non-pictorial map images>


## Training

* Adjust data_folder and log_folder in config.py
* Download training images and place them into the data folder
* Optionally adjust properties like models, number of runs, image input options in config.py (e.g. model_names = ["Xception"], run_nrs = ["1st"], maps_non_maps_crop_names = ["resize"])
* Run training.py to train the maps vs. non-maps classifier (uncomment lines in main method to train models to classify pictorial maps)


## Evaluation

* Run prediction.py to calculate scores for correctly classifying maps and non-maps (uncomment lines for pictorial and non-pictorial maps)
* Run evaluation.py to calculate accuracies and find wrong predictions (optionally adjust the threshold)
* Run evaluation_roc_curve.py to create a ROC curve plot from the predictions


## Source
https://github.com/tensorflow/tensorflow (Apache License, Copyright by The TensorFlow Authors)

#### Modifications
None