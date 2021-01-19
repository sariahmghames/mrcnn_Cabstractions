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



def get_orientation(result):
	rois = result["rois"]
	class_ids = result["class_ids"]
	scores = result["scores"]
	masks = result['masks']
	y = np.array([0,1])
	orientations = []
	print('rois are the bboxes in an image, of shape=', rois.shape)
	print('rois are the bboxes in image 0 =', rois[0])
	for ind in range(len(class_ids)):
		if class_ids[ind] == 3:
			x1, y1, x2, y2 = rois[ind]
			com = np.mean(masks[ind])
			std = np.std(masks[ind])
			extreme = com + std 
			orient = (extreme - com) / np.linalg.norm(extreme-com)
			alpha = y.dot(orient)
		else alpha = None
		orientations.append(alpha)
	return orientations


