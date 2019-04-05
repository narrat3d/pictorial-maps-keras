import argparse
import os
from helper_functions import listdir_fullpath
from tensorflow.python.keras.models import load_model
from PIL import Image
import config
from prediction import PREDICTION_METHODS
import shutil


def parse_args(args):
    parser = argparse.ArgumentParser(description='Classify images into maps and non-maps.')
    parser.add_argument('input_folder', type=str, help='Input folder with images.')
    parser.add_argument('output_folder', type=str, help='Folder where classified images will be copied.')
    args = parser.parse_args(args)
    
    if (not os.path.exists(args.input_folder)):
        raise Exception("The specified input folder does not exist.")
    
    if (not os.path.exists(args.output_folder)):
        raise Exception("The specified output folder does not exist.")
    
    return (args.input_folder, args.output_folder)


def classify(input_folder, output_folder, path_to_best_model, prediction_name, class_names):
    model = load_model(path_to_best_model)
    prediction_method = PREDICTION_METHODS[prediction_name]
    
    for class_name in class_names:
        class_output_folder = os.path.join(output_folder, class_name)
        os.mkdir(class_output_folder)
    
    for file_path in listdir_fullpath(input_folder):
        try :
            image = Image.open(file_path)
        except :
            continue
        
        score, _ = prediction_method(model, image, config.IMAGE_SIZE)
        
        if (score > 0.5):
            output_class = class_names[0]
        else :
            output_class = class_names[1]
            
        class_output_folder = os.path.join(output_folder, output_class)
        
        shutil.copy(file_path, class_output_folder)
    