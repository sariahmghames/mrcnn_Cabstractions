"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys

class KineticImportsFix:
	def __init__(self, kinetic_dist_packages="/opt/ros/kinetic/lib/python2.7/dist-packages"):
		self.kinetic_dist_packages = kinetic_dist_packages

	def __enter__(self):
		sys.path.remove(self.kinetic_dist_packages)

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.path.append(self.kinetic_dist_packages)


with KineticImportsFix():   
	import os
	import math
	import random
	import numpy as np
	import cv2
	import matplotlib.pyplot as plt
	

import tensorflow as tf

## To ensure tensorflow package is using your GPU
#sess = tf.Session()

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
CUSTOM_IMAGES_PATH = "/home/sariah/intProMP_franka/src/franka_gazebo/dataset/color1"


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
sys.path.append(ROOT_DIR)  # To find local version of the library
from shapes_cluster import ShapesConfig
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import add_utils as formulations

# Used only with ipython jupyter notebook
# %matplotlib inline


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = MODEL_DIR
DEFAULT_DATASET_YEAR = 2014



class InferenceConfig(ShapesConfig):
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model_inference = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "mask_rcnn_cluster_shapes_0020.h5")
#model_path = model_inference.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model_inference.load_weights(model_path, by_name=True)


print('start loading simulation images to infer on ')
Cimages = dataset_train.load_images_from_folder(CUSTOM_IMAGES_PATH)
print('finish loading simulation images to infer on ')
for cimg in Cimages:
	print('start testing on custom images')
	c_results = model_inference.detect([cimg], verbose=1)
	r = c_results[0]
	visualize.display_instances(cimg, r['rois'], r['masks'], r['class_ids'], 
                        dataset_val.class_names, r['scores'], ax=get_ax()) 