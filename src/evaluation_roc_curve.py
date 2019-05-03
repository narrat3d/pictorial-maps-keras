import json
import config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import get_prediction_label


model_names = config.model_names

task_name = config.maps_non_maps_task_name
prediction_names = config.maps_non_maps_prediction_names
class_names = config.maps_non_maps_class_names

"""
task_name = config.pictorial_maps_other_maps_task_name
class_names = config.pictorial_maps_other_maps_class_names
prediction_names = config.pictorial_maps_other_maps_prediction_names
"""
plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], 'k--')

roc_curves = []

for model_name in model_names:
    for prediction_name in prediction_names:
        prediction_file_path = config.get_predictions_file_path(task_name, model_name, prediction_name)
        data = json.load(open(prediction_file_path))
        
        y_true = []
        scores = []
        
        for class_name in class_names:
            y_true_value = (class_names.index(class_name) + 1) % 2
            
            for image_name, predictions in data[class_name].items():
                for prediction in predictions:
                    y_true.append(y_true_value)
                    scores.append(prediction["score"])

        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
        auc_ = auc(fpr, tpr)
        
        roc_curves.append({
            "model_name": model_name,
            "crop_name": get_prediction_label(prediction_name),
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_
        })


roc_curves.sort(key=lambda roc_curve: roc_curve["auc"], reverse=True)

for roc_curve in roc_curves:
    plt.plot(roc_curve["fpr"], roc_curve["tpr"], label='%s, %s (auc = {:.3f})'.format(roc_curve["auc"]) 
             % (roc_curve["model_name"], roc_curve["crop_name"]))


# plt.plot(fpr2, tpr2, label='Keras 2 (area = {:.3f})'.format(auc_rf2))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves: Pictorial maps vs. non-pictorial maps') # Pictorial maps vs. non-pictorial maps
plt.legend(loc='best')
plt.tight_layout()

axes = plt.gca()
axes.set_xlim([0,0.1]) # 0,0.3
axes.set_ylim([0.9,1.0]) # 0.7,1.0

plt.show()
plt.waitforbuttonpress()