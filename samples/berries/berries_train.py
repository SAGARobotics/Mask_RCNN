import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
BERRIES_DIR = os.path.join(ROOT_DIR, "samples/berries/ml_strawberry")


class BerryConfig(Config):
    """Configuration for training on berries dataset.
   
    """
    # Give the configuration a recognizable name
    NAME = "berries"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    

    STEPS_PER_EPOCH = 100

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 +2


class BerryDataset(utils.Dataset):
    

    def load_berry(self, dataset_dir, subset):
        
        self.add_class("berries",1, "calyx")
        self.add_class("berries",2, "flesh")

          # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset, "data")

        for filename in (os.listdir(dataset_dir)):
            image_path = os.path.join(dataset_dir, filename)
            image_id = os.path.basename(image_path)[:-4]
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

        

            self.add_image("berries", image_id=image_id, path=image_path, width=width, height=height)

    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        

        json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(info["path"]))), "json_files")
        json_id = self.image_reference(image_id)
       
        json_id = os.path.join(json_dir, json_id+".json")
        data = json.load(open(json_id))["labels"]

        #this might be a bug..
        #height, width  = data[0]["pixel"]["imagesize"][0], data[0]["pixel"]["imagesize"][1]
        
        mask = np.zeros([info["height"], info["width"], len(data)],dtype=np.uint8) 

        
        class_ids = []
        for i in range(len(data)):

            class_name = data[i]["category"][0][1]
            class_id = 1 if class_name == "calyx" else 2
            class_ids.append(class_id)

            #x = [round(pixel_pos["xnorm"] * 255.0) for pixel_pos in data[i]["polygon"]]
            #y = [round(pixel_pos["ynorm"] * 255.0) for pixel_pos in data[i]["polygon"]]

            y  = [pixel_pos[0] for pixel_pos in data[i]["pixel"]["regions"][0][0]]
            x  = [pixel_pos[1] for pixel_pos in data[i]["pixel"]["regions"][0][0]]
            rr, cc = skimage.draw.polygon(y,x)    
             
            mask[rr, cc, i] = 1
        
        return mask.astype(np.bool), np.array(class_ids)



    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["id"]
    

def train(model):
    """Train the model """

    dataset = "samples"
     # Training dataset.
    dataset_train = BerryDataset()
    dataset_train.load_berry(BERRIES_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BerryDataset()
    dataset_val.load_berry(BERRIES_DIR, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')




if __name__ == "__main__":
    config = BerryConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    weights_path = COCO_WEIGHTS_PATH
    if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    print("Loading weights ", weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    train(model)