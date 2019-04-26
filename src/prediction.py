import numpy as np
import os
import config
from tensorflow.python.keras.models import load_model
from PIL import Image
from training import random_crop, resize, middle_random_crop
from training import preprocess_image
import json
from tensorflow.python.keras import backend
import gc
from helper_functions import avg, crop_image_into_equal_regions




def predict(model, image):
    image_np = preprocess_image(image)
    
    predictions = model.predict(np.asarray([image_np]))
    scores = predictions[0]
    
    return scores[0].item()


def predict_regions(model, image, image_size):
    scores_of_first_class = []
    
    regions = crop_image_into_equal_regions(image, image_size)
    
    for region in regions: # upscale region if necessary
        image_region = resize(region["image"], image_size)
        first_class_score = predict(model, image_region)

        scores_of_first_class.append(first_class_score)
        
    return scores_of_first_class


def predict_on_resized_image(model, image, IMAGE_SIZE):
    resized_image = resize(image, IMAGE_SIZE)
    
    return predict(model, resized_image), None


def predict_on_middle_random_crop(model, image, IMAGE_SIZE):
    cropped_image, crop_values = middle_random_crop(image, IMAGE_SIZE)
    
    return predict(model, cropped_image), crop_values


def predict_on_random_crop(model, image, IMAGE_SIZE):
    cropped_image, crop_values = random_crop(image, IMAGE_SIZE)
    
    return predict(model, cropped_image), crop_values


def predict_on_grid_avg(model, image, IMAGE_SIZE):
    scores_of_first_class = predict_regions(model, image, IMAGE_SIZE) 
    first_class_scores_avg = avg(scores_of_first_class)
    
    return first_class_scores_avg, None


def predict_on_grid_max(model, image, IMAGE_SIZE):
    scores_of_first_class = predict_regions(model, image, IMAGE_SIZE) 
    first_class_scores_max = max(scores_of_first_class)
    
    return first_class_scores_max, None
    

PREDICTION_METHODS = {
    "resize": predict_on_resized_image,
    "middle_random_crop": predict_on_middle_random_crop,
    "random_crop": predict_on_random_crop,
    "avg_over_gridded_crops": predict_on_grid_avg,
    "max_in_gridded_crops": predict_on_grid_max
}


if __name__ == '__main__':
    model_names = config.model_names
    run_nrs = config.run_nrs
    
    eval_image_folder = config.maps_non_maps_eval_image_folder
    task_name = config.maps_non_maps_task_name
    class_names = config.maps_non_maps_class_names
    prediction_names = config.maps_non_maps_prediction_names
    get_crop_for_prediction_names = config.get_maps_non_maps_crop_for_prediction_name
    
    """
    eval_image_folder = config.pictorial_maps_other_maps_eval_image_folder
    task_name = config.pictorial_maps_other_maps_task_name
    class_names = config.pictorial_maps_other_maps_class_names
    prediction_names = config.pictorial_maps_other_maps_prediction_names
    crop_names = config.pictorial_maps_other_maps_crop_names
    get_crop_for_prediction_names = config.get_pictorial_maps_other_maps_crop_for_prediction_name
    """
    
    for model_name in model_names:
        print(model_name)
        
        for prediction_name in prediction_names:
            print(prediction_name)
            
            prediction_method = PREDICTION_METHODS[prediction_name]
            output_file_path = config.get_predictions_file_path(task_name, model_name, prediction_name)
            
            predictions = {}
            
            for run_nr in run_nrs:
                print(run_nr)
                
                crop_name = get_crop_for_prediction_names(prediction_name)
                training_log_folder = config.get_training_log_folder(task_name, run_nr)
                temp_model_name = config.get_temp_model_name(model_name, task_name, crop_name)
                model_path = os.path.join(training_log_folder, temp_model_name)
                     
                model = load_model(model_path)
                
                for class_name in class_names:
                    print(class_name)
                    
                    image_input_folder = os.path.join(eval_image_folder, class_name)
                    
                    prediction_for_class = predictions.setdefault(class_name, {})
                    
                    for image_name in os.listdir(image_input_folder):
                        image_path = os.path.join(image_input_folder, image_name)
                        
                        try :
                            image = Image.open(image_path)
                        except :
                            continue
                        
                        score, crop_values = prediction_method(model, image, config.IMAGE_SIZE)
                        
                        image_predictions = prediction_for_class.setdefault(image_name, [])
                        
                        image_predictions.append({
                            "score": score,
                            "crop": crop_values
                        })
                        
                del model
                backend.clear_session()
                gc.collect()
                
            json.dump(predictions, open(output_file_path, "w"))