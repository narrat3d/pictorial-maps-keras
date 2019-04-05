import sys
from classification import parse_args, classify
import config


def main(args=None):
    if args is None:
        args = sys.argv[1:]
        
    (input_folder, output_folder) = parse_args(args)
    
    classify(input_folder, output_folder, 
             config.pictorial_maps_other_maps_best_model["path"],
             config.pictorial_maps_other_maps_best_model["prediction_name"],
             config.pictorial_maps_other_maps_class_names)

if __name__ == '__main__':
    main()