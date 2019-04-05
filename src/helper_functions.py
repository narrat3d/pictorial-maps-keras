import os
import shutil
import math
from PIL import Image


def mkdir_if_not_exists(path):
    return os.makedirs(path, exist_ok=True)
        

def clear_and_mkdir(file_path):
    if (os.path.exists(file_path)):
        shutil.rmtree(file_path)
        
    os.mkdir(file_path)


# source: https://stackoverflow.com/questions/120656/directory-tree-listing-in-python
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def avg(list_):
    return sum(list_) / len(list_)


def crop_image_into_equal_regions(image, cropped_image_size):
    regions = []

    image_width, image_height = image.size

    column_number = math.ceil(image_width / cropped_image_size)
    row_number = math.ceil(image_height / cropped_image_size)

    region_width = image_width / column_number
    region_height = image_height / row_number

    for j in range(row_number):
        for i in range(column_number):
            region_mid_x = i * region_width + region_width / 2
            region_mid_y = j * region_height + region_height / 2

            x_min = round(region_mid_x - cropped_image_size / 2)
            x_min = max(0, x_min)
            y_min = round(region_mid_y - cropped_image_size / 2)
            y_min = max(0, y_min)

            x_max = min(image_width, x_min + cropped_image_size)
            y_max = min(image_height, y_min + cropped_image_size)

            x_min = max(x_max - cropped_image_size, 0)
            y_min = max(y_max - cropped_image_size, 0)

            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            regions.append({
                "x": i,
                "y": j,
                "image": cropped_image
            })

    return regions


def get_image_paths(input_folders):
    all_image_paths = []

    for input_folder in input_folders:
        image_paths = listdir_fullpath(input_folder)
        all_image_paths.extend(image_paths)

    return all_image_paths


def get_average_image_size(image_paths):
    widths = 0
    heights = 0
    image_number = 0

    for image_path in image_paths:
        try:
            image = Image.open(image_path)
        except Exception:
            continue

        widths += image.size[0]
        heights += image.size[1]
        image_number += 1

    return (widths / len(image_paths), heights / len(image_paths))


if __name__ == '__main__':
    import config
    # input_folders = config.get_image_folders(config.maps_non_maps_task_name, config.maps_non_maps_class_names)
    input_folders = config.get_image_folders(config.pictorial_maps_other_maps_task_name, config.pictorial_maps_other_maps_class_names)

    input_paths = get_image_paths(input_folders)
    result = get_average_image_size(input_paths)

    print(result)
