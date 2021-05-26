import imp
import os
#from samples.berries.berries import BERRIES_DIR
import sys
ROOT_DIR = os.path.abspath("../../")
#sys.path.insert(0, "/home/temi/Mask_RCNN")
sys.path.insert(0, ROOT_DIR)
import mrcnn
import json
import datetime
from imgaug.imgaug import is_float_array
import numpy as np
from numpy.lib.type_check import real_if_close
import skimage.draw
import mrcnn.model as modellib
import cv2
import random
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize

from berries_train import BerryConfig, BerryDataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("weights", help="paths to weights file")
parser.add_argument("data", help="path to directory containing train, val, test")
parser.add_argument("results", help="path to store result images")
args = parser.parse_args()

MODEL_PATH = args.weights
BERRIES_DIR = args.data
OUTPUT_FOLDER = args.results

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class BerryConfigInference(BerryConfig):
    GPU_COUNT =1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_CONFIDENCE = 0.9


def run_eval(dataset, show_image=False):
    APs = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        results = model.detect([image], verbose=1)
        r = results[0]

        #import pdb; pdb.set_trace()

        """
        AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
        
        APs.append(AP)
        print (AP, "-", precisions, "-", recalls, "-", overlaps)
        """
        
        _, ax = plt.subplots(1, figsize=(16,16))

        AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])

        #print (AP)
        APs.append(AP)
        
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], figsize=(8,8),ax=ax)
    
        #save image ..
        filename = dataset.image_reference(image_id)
        
        plt.savefig(os.path.join(OUTPUT_FOLDER, filename+".png"))
        if show_image:
            plt.show()
    return APs

if __name__=="__main__":

    inference_config = BerryConfigInference()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)

    # Load trained weights
    print("Loading weights from ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)

    #load data...

    dataset_val = BerryDataset()
    dataset_val.load_berry(BERRIES_DIR, "val")
    dataset_val.prepare()

    APs = run_eval(dataset_val)
    
    print("Mean AP over {} images: {:.3f}".format(len(APs), np.mean(APs)))



    

