import os
import gc
import random
import numpy as np
import config
from PIL import Image, ImageOps
# do not remove
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
from tensorflow.python import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger,\
    ModelCheckpoint
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.saving import load_model


def random_crop(image, desired_image_size):
    image = upscale_eventually(image, desired_image_size)
    
    width = image.size[0]
    height = image.size[1]
    
    left = random.randint(0, width - desired_image_size + 1)
    upper = random.randint(0, height - desired_image_size + 1)    
    right = left + desired_image_size
    lower = upper + desired_image_size
    
    crop_values = (left, upper, right, lower)
    cropped_image = image.crop(crop_values)
    
    return cropped_image, crop_values


def middle_random_crop(image, desired_image_size):
    image = upscale_eventually(image, desired_image_size)
    
    width = image.size[0]
    height = image.size[1]
    
    if (width < height):
        left = 0
        upper = random.randint(0, height - width + 1)
        right = left + width
        lower = upper + width
    else :
        left = random.randint(0, width - height + 1)   
        upper = 0
        right = left + height  
        lower = upper + height
    
    crop_values = (left, upper, right, lower)
    cropped_image = image.crop(crop_values)
    resized_image = resize(cropped_image, desired_image_size)
    
    return resized_image, crop_values


def upscale_eventually(image, minimum_image_size):
    width = image.size[0]
    height = image.size[1]
    
    if (width < height):
        if (width < minimum_image_size):
            upscaling_ratio = minimum_image_size/width
            upscaled_height = round(upscaling_ratio * height)
            image = image.resize((minimum_image_size, upscaled_height), Image.LANCZOS)
    else :
        if (height < minimum_image_size):
            upscaling_ratio = minimum_image_size/height
            upscaled_width = round(upscaling_ratio * width) 
            image = image.resize((upscaled_width, minimum_image_size), Image.LANCZOS)  
            
    return image


def resize(image, desired_image_size):
    resized_image = image.resize((desired_image_size, desired_image_size), Image.LANCZOS)
    
    return resized_image


# is already cropped to the input size
def manual_crop(image, desired_image_size):
    image = image.resize((desired_image_size, desired_image_size), Image.LANCZOS)
    
    return image, None


def get_random_file_names_for_manual_crop(input_folder, class_names):
    file_names = []
    
    for class_name in class_names:
        subfolder_path = os.path.join(input_folder, class_name)
        
        file_names_for_class = os.listdir(subfolder_path)
        
        file_names_for_root = {}
        
        for file_name in file_names_for_class:
            file_name_root = file_name.split("___")[0]
            file_names_with_this_root = file_names_for_root.setdefault(file_name_root, [])
            file_names_with_this_root.append(file_name)
        
        for file_names_with_one_root in file_names_for_root.values():
            random.shuffle(file_names_with_one_root)
            random_file_name = file_names_with_one_root.pop()

            random_file_name_with_class = os.path.join(class_name, random_file_name)
            file_names.append(random_file_name_with_class)
            
    return file_names


def preprocess_image(image):
    image_np = np.asarray(image, dtype=np.float32)
    
    if (len(image_np.shape) == 2):
        # source: https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
        image_np = np.stack((image_np,)*3, -1)
    
    image_np = image_np[:, :, 0:3]
    standardized_image = imagenet_utils.preprocess_input(image_np)
    
    return standardized_image


class DataGenerator(Sequence):

    def __init__(self, mode, input_folder, class_names, batch_size, img_size, cropping_method):
        self.mode = mode
        self.input_folder = os.path.join(input_folder, mode)
        self.batch_size = batch_size
        self.img_size = img_size
        self.class_names = class_names
        self.file_names = []
        self.cropping_method = cropping_method
        
        if (self.cropping_method != manual_crop):
            # file names are the same for each epoch
            for class_name in class_names:
                subfolder_path = os.path.join(self.input_folder, class_name)
                
                file_names_for_class = os.listdir(subfolder_path)
                file_names_with_class = map(lambda file_name: os.path.join(class_name, file_name), file_names_for_class)
                
                self.file_names.extend(list(file_names_with_class))
        
        self.on_epoch_end()

    'Denotes the number of batches per epoch'
    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        file_names_batch = self.file_names[index * self.batch_size : index * self.batch_size + self.batch_size]
        X, y = self.__data_generation(file_names_batch)

        return X, y

    def on_epoch_end(self):
        if (self.cropping_method == manual_crop):
            self.file_names = get_random_file_names_for_manual_crop(self.input_folder, self.class_names)
        
        random.shuffle(self.file_names)

    def __data_generation(self, image_file_paths):
        source = np.empty((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        target = np.zeros((self.batch_size, 2), dtype=np.float32)

        for i, image_file_path in enumerate(image_file_paths):
            image_input_path = os.path.join(self.input_folder, image_file_path)
            
            try :
                image = Image.open(image_input_path)
            except Exception:
                print ("Could not open %s." % image_input_path)
                continue

            cropped_image = self.cropping_method(image, self.img_size)
            
            if (isinstance(cropped_image, ().__class__)):
                # cropping values not needed
                image = cropped_image[0]
            else :
                image = cropped_image
            
            if (self.mode == "train" and random.random() < 0.5):
                image = ImageOps.mirror(image)
            
            source[i,] = preprocess_image(image)
            
            current_class_name = image_file_path.split(os.sep)[0]
            boolean_class_names = list(map(lambda class_name: class_name == current_class_name, self.class_names))
            
            target[i, ] = np.asarray(boolean_class_names, np.float32)
         
        return source, target


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        accuracy = logs.get('acc')
        self.acc.append(accuracy)


def train(classification_model_name, task_name, crop_name, image_input_folder, log_folder):
    class_names = task_name.split("_vs_")
    
    classification_model_class = eval(classification_model_name)
    cropping_method = eval(crop_name)
    
    log_file_name = "%s_%s_%s_%sep_%slr" % (classification_model_name, task_name, crop_name, config.EPOCHS, config.LEARNING_RATE)

    csv_log_file_path = os.path.join(log_folder, log_file_name + ".csv")
    model_file_path = os.path.join(log_folder, log_file_name + ".h5")
 
    datagen = DataGenerator("train", image_input_folder, class_names, config.BATCH_SIZE, config.IMAGE_SIZE, cropping_method)
    validation_generator = DataGenerator("eval", image_input_folder, class_names, config.BATCH_SIZE, config.IMAGE_SIZE, cropping_method) 
    
    classification_model = classification_model_class(weights='imagenet', include_top=False, pooling='avg')
    
    dense_layer = Dense(len(class_names), activation='softmax', name='predictions')(classification_model.output)
    """
    model = load_model(model_file_path)
    """
    model = Model(classification_model.input, dense_layer)
    
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
              metrics=['accuracy']) # accuracy is the same as categorical_accuracy
    
    tensorboard_callback = TensorBoard(log_dir=log_folder)
    
    csv_logger = CSVLogger(csv_log_file_path)
    
    history = AccuracyHistory()  
    
    model_checkpoint = ModelCheckpoint(model_file_path, save_weights_only=False, 
                                       monitor='val_acc', mode="max", 
                                       verbose=1, save_best_only=True)
    
    model.fit_generator(datagen,
              validation_data=validation_generator,
              epochs=config.EPOCHS,
              callbacks=[history, tensorboard_callback, csv_logger, model_checkpoint])
    
    del model

    backend.clear_session()
    gc.collect()


if __name__ == '__main__':
    task_name = config.maps_non_maps_task_name
    image_input_folder = config.maps_non_maps_image_folder
    crop_names = config.maps_non_maps_crop_names
    
    """
    task_name = config.pictorial_maps_other_maps_task_name
    image_input_folder = config.pictorial_maps_other_maps_image_folder
    crop_names = config.pictorial_maps_other_maps_crop_names
    """

    """
    task_name = config.pictorial_maps_other_maps_task_name
    image_input_folder = config.pictorial_maps_other_maps_cropped_image_folder
    crop_names = config.pictorial_maps_other_maps_manual_crop_names
    """
    
    for run_nr in config.run_nrs:
        log_folder = config.get_training_log_folder(task_name, run_nr)
        
        for classification_model_name in config.model_names:
            for crop_name in crop_names: 
                train(classification_model_name, task_name, crop_name, image_input_folder, log_folder)
