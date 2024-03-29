# Classification of (non-)maps and (non-)pictorial maps

This is code for the article [Detection of Pictorial Map Objects with Convolutional Neural Networks](https://doi.org/10.1080/00087041.2020.1738112). Visit the [project website](http://narrat3d.ethz.ch/detection-of-pictorial-map-objects-with-cnns/) for more information.

![maps](https://github.com/narrat3d/pictorial-maps-keras/assets/9949879/4b9bcb69-4ff0-4f59-9d5e-7f29b82e2cfc)

Image sources: [Physical Map of the World](https://commons.wikimedia.org/wiki/File:Weltkarte.jpg), [Tampa-Bay Aerial View Map](https://commons.wikimedia.org/wiki/File:Tampa-Bay-aerial-map.png)

## Installation

* Requires [Python 3.6.x](https://www.python.org/downloads/)
* Requires [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
* Download [this project](https://gitlab.ethz.ch/sraimund/pictorial-maps-keras/-/archive/master/pictorial-maps-keras-master.zip)
* pip install -r requirements.txt


## Inference

* Adjust models_folder in config.py
* Download [trained models](https://ikgftp.ethz.ch/?u=uTyy&p=7dbt&path=/pictorial_maps_keras_models.zip) and place them inside the models folder

#### Maps vs. Non-maps
* Run classify_maps.py \<input folder with images> \<output folder for map and non-map images>

#### Pictorial maps vs. non-pictorial maps
* Run classify_pictorial_maps.py \<input folder with map images> \<output folder for pictorial map and non-pictorial map images>


## Training

* Adjust data_folder and log_folder in config.py
* Download [training data](https://ikgftp.ethz.ch/?u=bFup&p=fR7C&path=/pictorial_maps_keras_data.zip) and place them into the data folder
* Optionally adjust properties like models (e.g. model_names = ["Xception"]), number of runs (e.g. run_nrs = ["1st"]), image input options (e.g. maps_non_maps_crop_names = ["resize"]) in config.py
* Run training.py to train the maps vs. non-maps classifier (uncomment lines in main method to train models to classify pictorial maps)


## Evaluation

* Run prediction.py to calculate scores for correctly classifying maps and non-maps (uncomment lines for pictorial and non-pictorial maps)
* Run evaluation.py to calculate accuracies and find wrong predictions (optionally adjust the threshold)
* Run evaluation_roc_curve.py to create a ROC curve plot from the predictions

## Citation

Please cite the following article when using this code:
```
@article{schnuerer2021detection,
  author = {Raimund Schnürer, René Sieber, Jost Schmid-Lanter, A. Cengiz Öztireli and Lorenz Hurni},
  title = {Detection of Pictorial Map Objects with Convolutional Neural Networks},
  journal = {The Cartographic Journal},
  volume = {58},
  number = {1},
  pages = {50-68},
  year = {2021},
  doi = {10.1080/00087041.2020.1738112}
}
```

© 2019-2020 ETH Zurich, Raimund Schnürer
