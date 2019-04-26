import os

# folder where training data is located
data_folder = r"E:\CNN\classification"

# folder where trained models and prediction results will be stored
log_folder = r"E:\CNN\logs\classification"

# folder where best models are stored
models_folder = r"E:\CNN\models\keras"


run_nrs = ["1st", "2nd", "3rd"]

IMAGE_SIZE = 299
LEARNING_RATE = 0.00001
EPOCHS = 40
BATCH_SIZE = 16

model_names = ["Xception", "InceptionResNetV2"]
train_folder_name = "train"
eval_folder_name = "eval"

maps_non_maps_class_names = ["maps", "non_maps"]
maps_non_maps_task_name = "_vs_".join(maps_non_maps_class_names)
maps_non_maps_crop_names = ["resize", "middle_random_crop", "random_crop"]  
maps_non_maps_prediction_names = ["resize", "middle_random_crop", "random_crop", "avg_over_gridded_crops"]

get_maps_non_maps_crop_for_prediction_name = lambda prediction_name: {
    "resize": "resize",
    "middle_random_crop": "middle_random_crop",
    "random_crop": "random_crop",
    "avg_over_gridded_crops": "random_crop"
}[prediction_name]

pictorial_maps_other_maps_class_names = ["pictorial_maps", "other_maps"]
pictorial_maps_other_maps_task_name = "_vs_".join(pictorial_maps_other_maps_class_names)
pictorial_maps_other_maps_crop_names = ["resize"]
pictorial_maps_other_maps_manual_crop_names = ["manual_crop"]

pictorial_maps_other_maps_prediction_names = ["resize", "random_crop", "avg_over_gridded_crops", "max_in_gridded_crops"]

get_pictorial_maps_other_maps_crop_for_prediction_name = lambda prediction_name: {
    "resize": "resize",
    "random_crop": "manual_crop",
    "avg_over_gridded_crops": "manual_crop",
    "max_in_gridded_crops": "manual_crop"
}[prediction_name]


maps_non_maps_image_folder = os.path.join(data_folder, maps_non_maps_task_name)
maps_non_maps_eval_image_folder = os.path.join(maps_non_maps_image_folder, eval_folder_name)
pictorial_maps_other_maps_image_folder = os.path.join(data_folder, pictorial_maps_other_maps_task_name)
pictorial_maps_other_maps_eval_image_folder = os.path.join(pictorial_maps_other_maps_image_folder, eval_folder_name)
pictorial_maps_other_maps_cropped_image_folder = os.path.join(data_folder, pictorial_maps_other_maps_task_name + "_cropped")

get_training_log_folder = lambda task_name, run_nr: os.path.join(log_folder, "%s_%s_run") % (task_name, run_nr)

get_predictions_log_folder = lambda task_name: os.path.join(log_folder, "%s_predictions") % task_name
get_predictions_file_name = lambda model_name, prediction_name: "predictions_%s_%s.json" % (model_name, prediction_name)
get_predictions_file_path = lambda task_name, model_name, prediction_name: \
    os.path.join(get_predictions_log_folder(task_name), get_predictions_file_name(model_name, prediction_name))

get_predictions_wrong_folder = lambda task_name: os.path.join(log_folder, "%s_predictions_wrong") % task_name

get_temp_model_name = lambda model_name, task_name, crop_name: "%s_%s_%s_40ep_1e-05lr.h5" % (model_name, task_name, crop_name)

""" 
maps_non_maps_best_model = { # faster
    "path": os.path.join(models_folder, "InceptionResNetV2_maps_vs_non_maps_resize.h5"),
    "prediction_name": "resize"
}
"""
maps_non_maps_best_model = { # better
    "path": os.path.join(models_folder, "Xception_maps_vs_non_maps_random_crop.h5"),
    "prediction_name": "avg_over_gridded_crops"
}

"""
pictorial_maps_other_maps_best_model = { # faster
    "path": os.path.join(models_folder, "Xception_pictorial_maps_vs_other_maps_resize.h5"),
    "prediction_name": "resize"
}
"""
pictorial_maps_other_maps_best_model = { # better
    "path": os.path.join(models_folder, "Xception_pictorial_maps_vs_other_maps_manual_crop.h5"),
    "prediction_name": "avg_over_gridded_crops"
}

def get_image_folders(task_name, class_names):
    image_folders = []
    root_folder = os.path.join(data_folder, task_name)

    for class_name in class_names:
        image_folders.append(os.path.join(root_folder, train_folder_name, class_name))
        image_folders.append(os.path.join(root_folder, eval_folder_name, class_name))

    return image_folders
