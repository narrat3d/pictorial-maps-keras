from collections import Counter
import json
import shutil
import config
import os
from helper_functions import mkdir_if_not_exists

wrong_prediction_threshold = 11
model_names = config.model_names

task_name = config.maps_non_maps_task_name
image_input_folder = config.maps_non_maps_eval_image_folder
prediction_names = config.maps_non_maps_prediction_names
class_names = config.maps_non_maps_class_names

"""
task_name = config.pictorial_maps_other_maps_task_name
image_input_folder = config.pictorial_maps_other_maps_eval_image_folder
prediction_names = config.pictorial_maps_other_maps_prediction_names
class_names = config.pictorial_maps_other_maps_class_names
"""

incorrect_images = {}
accuracies = {}

for class_name in class_names:
    incorrect_images[class_name] = []

for model_name in model_names:
    print(model_name)
    
    for prediction_name in prediction_names:
        prediction_file_path = config.get_predictions_file_path(task_name, model_name, prediction_name)
        data = json.load(open(prediction_file_path))
        
        results = {
            "true": {
                "positives": 0,
                "negatives": 0
            },
            "false": {
                "positives": 0,
                "negatives": 0                
            }
        }
         
        for class_index, class_name in enumerate(class_names):
            if (class_index == 0):
                is_prediction_correct = lambda score: score > 0.5
                result_type = "positives"
            else :
                is_prediction_correct = lambda score: score < 0.5
                result_type = "negatives"
            
            for image_name, predictions in data[class_name].items():
                for prediction in predictions: # contains the predictions of three runs
                    if (is_prediction_correct(prediction["score"])):
                        results["true"][result_type] += 1
                    else :
                        results["false"][result_type] += 1
                        incorrect_images[class_name].append(image_name)
        
        print(prediction_name)
        correct_results = results["true"]["positives"] + results["true"]["negatives"]
        incorrect_results = results["false"]["positives"] + results["false"]["negatives"]
        accuracy = correct_results / (correct_results + incorrect_results)
        
        precision = results["true"]["positives"] / (results["true"]["positives"] + results["false"]["positives"])
        recall = results["true"]["positives"] / (results["true"]["positives"] + results["false"]["negatives"])
        f1_score = 2 * precision * recall / (precision + recall)
        
        print("Accuracy: ", round(accuracy * 100, 2))
        print("Precision: ", round(precision * 100, 2))
        print("Recall: ", round(recall * 100, 2))
        print("F1 score: ", round(f1_score * 100, 2))
        print()
"""
for class_name in class_names: 
    occurences = Counter(incorrect_images[class_name])
    
    for image_name, number_of_occurences in occurences.items():
        if (number_of_occurences >= wrong_prediction_threshold):
            image_output_folder = os.path.join(config.get_predictions_wrong_folder(task_name), class_name)
            mkdir_if_not_exists(image_output_folder)
            input_image_path = os.path.join(image_input_folder, class_name, image_name)
            shutil.copy(input_image_path, image_output_folder)
"""