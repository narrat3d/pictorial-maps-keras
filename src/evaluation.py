from collections import Counter
import json
import shutil
import config
import os

wrong_prediction_threshold = 11
model_names = config.model_names
"""
task_name = config.maps_non_maps_task_name
image_input_folder = config.maps_non_maps_eval_image_folder
prediction_names = config.maps_non_maps_prediction_names
class_names = config.maps_non_maps_class_names

"""
task_name = config.pictorial_maps_other_maps_task_name
image_input_folder = config.pictorial_maps_other_maps_eval_image_folder
prediction_names = config.pictorial_maps_other_maps_prediction_names
class_names = config.pictorial_maps_other_maps_class_names


incorrect_images = {}
accuracies = {}

for class_name in class_names:
    incorrect_images[class_name] = []

for model_name in model_names:
    print(model_name)
    
    for prediction_name in prediction_names:
        prediction_file_path = config.get_predictions_file_path(task_name, model_name, prediction_name)
        data = json.load(open(prediction_file_path))
        
        correct_detections = 0
        counter = 0
         
        for class_index, class_name in enumerate(class_names):
            if (class_index == 0):
                is_prediction_correct = lambda score: score > 0.5
            else :
                is_prediction_correct = lambda score: score < 0.5
            
            for image_name, predictions in data[class_name].items():
                for prediction in predictions:
                    if (is_prediction_correct(prediction["score"])):
                        correct_detections += 1 
                    else :
                        incorrect_images[class_name].append(image_name)
    
                    counter += 1
        
        print(prediction_name)
        print(round(correct_detections / counter * 100, 2))

for class_name in class_names: 
    # print(class_name)
    occurences = Counter(incorrect_images[class_name])
    
    for image_name, number_of_occurences in occurences.items():
        if (number_of_occurences >= wrong_prediction_threshold): 
            # print(image_name)
            image_output_folder = os.path.join(config.get_predictions_wrong_folder(task_name), class_name)
            input_image_path = os.path.join(image_input_folder, class_name, image_name)
            shutil.copy(input_image_path, image_output_folder)