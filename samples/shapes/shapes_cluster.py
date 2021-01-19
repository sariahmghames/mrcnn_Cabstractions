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
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import add_utils as formulations
import shapes_utils 

# Used only with ipython jupyter notebook
# %matplotlib inline


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = MODEL_DIR
DEFAULT_DATASET_YEAR = 2014


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "cluster_shapes"

	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 0
	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + 4  # background + 4 shapes

	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	IMAGE_MIN_DIM = 1024 #128
	IMAGE_MAX_DIM = 1920

	# Use smaller anchors because our image and objects are small
	## Sariah comment out the following for cpu usage
	#RPN_ANCHOR_SCALES = (8, 16,32,64)  # anchor side in pixels #64

	# Reduce training ROIs per image because the images are small and have
	# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 32

	# Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 1000  # 500

	# use small validation steps since the epoch is small
	VALIDATION_STEPS = 500


class ShapesDataset(utils.Dataset):
	"""Generates the shapes synthetic dataset. The dataset consists of simple
	shapes (triangles, squares, circles, rectangles) placed randomly on a blank surface.
	The images are generated on the fly. No file access required.
	"""

	def load_shapes(self, count, height, width):
		"""Generate the requested number of synthetic images.
		count: number of images to generate.
		height, width: the size of the generated images.
		"""
		# Add classes
		#self.add_class("shapes", 1, "square") # source, class_id, class_name
		self.add_class("shapes", 1, "circle")
		self.add_class("shapes", 2, "triangle")
		#self.add_class("shapes", 4, "rectangle")
		self.add_class("shapes", 3, "stems")
		self.add_class("shapes", 4, "tabletop")

		self.num_shapes = 10
		self.mPA_train = False

		# Add images
		# Generate random specifications of images (i.e. color and
		# list of shapes sizes and locations). This is more compact than
		# actual images. Images are generated on the fly in load_image().
		for i in range(count):
			bg_color, shapes = self.random_image(height, width)
			self.add_image("shapes", image_id=i, path=None,
							width=width, height=height,
							bg_color=bg_color, shapes=shapes)
			if self.mPA_train:
				image_save = self.load_image_to_folder(image_id = i)

	
	def load_image(self, image_id):
		"""Generate an image from the specs of the given image ID.
		Typically this function loads the image from a file, but
		in this case it generates the image on the fly from the
		specs in image_info.
		"""
		info = self.image_info[image_id]
		bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
		image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
		image = image * bg_color.astype(np.uint8)
		for shape, color, dims, theta in info['shapes']:
			image = self.draw_shape(image, shape, dims, color, theta)

		return image


	def load_image_to_folder(self, image_id):
		"""Generate an image from the specs of the given image ID.
		Typically this function loads the image from a file, but
		in this case it generates the image on the fly from the
		specs in image_info.
		"""
		info = self.image_info[image_id]
		bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
		image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
		image = image * bg_color.astype(np.uint8)
		for shape, color, dims, theta in info['shapes']:
			image = self.draw_shape(image, shape, dims, color, theta)
			cv2.imwrite(os.path.join(ROOT_DIR, 'shapes_geneation' , str(image_id),'.jpg'), image)
		return image


	def load_test_image(self,):
	    """Load simulation recorded images """


	def load_images_from_folder(self, folder):
		images = []
		print('start loading custom images')
		print('folder of custom images=', folder)
		for filename in os.listdir(folder):
		    img = cv2.imread(os.path.join(folder,filename))
		    if img is not None:
		        images.append(img)
		        print('len of custom images=', len(images))
		return images


	def image_reference(self, image_id):
	    """Return the shapes data of the image."""
	    info = self.image_info[image_id]
	    if info["source"] == "shapes":
	        return info["shapes"]
	    else:
	        super(self.__class__).image_reference(self, image_id)



	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID.
		"""
		info = self.image_info[image_id]
		shapes = info['shapes']
		count = len(shapes)
		mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
		for i, (shape, _, dims, theta) in enumerate(info['shapes']):
			mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
												shape, dims, 1, theta)
		# Handle occlusions
		occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
		for i in range(count - 2, -1, -1):
		    mask[:, :, i] = mask[:, :, i] * occlusion
		    occlusion = np.logical_and(
		        occlusion, np.logical_not(mask[:, :, i]))
		# Map class names to class IDs.
		class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
		return mask, class_ids.astype(np.int32)



	def draw_shape(self, image, shape, dims, color, angle):  
		"""Draws a shape from the given specs."""
		# Get the center x, y and the size s
		x, y, s, sg, ss, st = dims
		if shape == 'square':
			image = formulations.draw_angled_rec(x, y, image, s, s, color, angle)
			#image = cv2.rectangle(image, (x - s, y - s),(x + s, y + s), color, -1)
		elif shape == 'stems' :  # and (color in self.ColorS)
			image = formulations.draw_angled_rec(x, y, image, s, ss, self.ColorS, angle)
			#image = cv2.rectangle(image, (x - ss, y - s),(x + ss, y + s), color, -1)
		elif shape == 'tabletop': #and (color in self.ColorT):
			#image = cv2.rectangle(image, (x - s, y - sg),(x + s, y + sg), self.ColorT, -1)
			image = formulations.draw_angled_rec(x, y, image, s, st, self.ColorT, 0)
		#elif shape == 'rectangle' : # and (color in self.ColorS)
			#shape = 'stems'
			#image = formulations.draw_angled_rec(x, y, image, ss, s, color, angle)
			#image = cv2.rectangle(image, (x - ss, y - s), (x + ss, y + s), color, -1)
		elif shape == 'rectangle' and (color in self.ColorT):
			shape = 'tabletop'
			image = cv2.rectangle(image, (x - st, y - s), (x + st, y + s), color, -1)
		elif shape == 'rectangle':
			image = formulations.draw_angled_rec(x, y, image, sg, s, color, angle)
			#image = cv2.rectangle(image, (x - sg, y - s),(x + sg, y + s), color, -1)
		# elif shape=='stems':
		# 	image = cv2.rectangle(image, (x - sg, y - s),(x + sg, y + s), color, -1)
		# elif shape == 'tabletop':
		# 	image = cv2.rectangle(image, (x - sg, y - s),(x + sg, y + s), color, -1)
		elif shape == "circle":
			image = cv2.circle(image, (x, y), int(s), color, -1)
		elif shape == "triangle":
			points = np.array([[(x, y - s),
								(x - s / math.sin(math.radians(60)), y + s),
								(x + s / math.sin(math.radians(60)), y + s),
								]], dtype=np.int32)
			image = cv2.fillPoly(image, points, color)

		return image

    
	def random_shape(self, height, width):
		"""Generates specifications of a random shape that lies within
		the given height and width boundaries.
		Returns a tuple of three valus:
		* The shape name (square, circle, ...)
		* Shape color: a tuple of 3 values, RGB.
		* Shape dimensions: A tuple of values that define the shape size
		                    and location. Differs per shape type.
		"""
		# Shape
		shapeT = random.choice(["square", "circle", "triangle", "rectangle", "stems", "tabletop"])
		shape = random.choice(["circle", "triangle", "stems", "tabletop"])
		# Color
		Color = tuple([random.randint(0, 255) for _ in range(3)])
		self.ColorT = random.choice([tuple([105,105,105]), tuple([128,128,128]), tuple([169,169,169]), tuple([192,192,192]), tuple([211,211,211]),tuple([220,220,220])])
		self.ColorS = random.choice([tuple([124,252,0]), tuple([127,255,0]), tuple([50,205,50]), tuple([0,255,0]), tuple([34,139,34]),tuple([0,128,0]), tuple([0,100,0]), tuple([173,255,47]), tuple([154,205,50]), tuple([0,255,127]), tuple([0,250,154]), tuple([144,238,144]), tuple([152,251,152]), tuple([143,188,143]), tuple([60,179,113]), tuple([32,178,170]), tuple([46,139,87]), tuple([128,128,0]), tuple([85,107,47]), tuple([107,142,35])])
		ColorT_bgr = random.choice([tuple([105,105,105]), tuple([128,128,128]), tuple([169,169,169]), tuple([192,192,192]), tuple([211,211,211]),tuple([220,220,220])])
		ColorS_bgr = random.choice([tuple([0,252,124]), tuple([0,255,127]), tuple([50,205,50]), tuple([0,255,0]), tuple([34,139,34]),tuple([0,128,0]), tuple([0,100,0]), tuple([47,255,173]), tuple([50,205,154]), tuple([127,255,0]), tuple([154,250,0]), tuple([144,238,144]), tuple([152,251,152]), tuple([143,188,143]), tuple([113,179,60]), tuple([170,178,32]), tuple([87,139,46]), tuple([0,128,128]), tuple([47,107,85]), tuple([35,142,107])])

		color = random.choice([Color, self.ColorT, self.ColorS])
		# Center x, y
		buffer = 5
		y = random.randint(buffer, height - buffer - 1)
		x = random.randint(buffer, width - buffer - 1)
		# Size
		s = (random.randint(buffer, height // 5))*4
		sg = (random.randint(buffer, height // 5))*3
		#ss = random.uniform(0.01, 0.1)
		#st = random.uniform(15, 50)
		ss = random.uniform(0.2, 0.8)
		st = random.uniform(80,200)
		theta = random.uniform(0, 180)
		#return shape, color, (x, y, s, sg, ss, st), theta
		return shape, Color, (x, y, s, sg, ss, st), theta


	def random_image(self, height, width):
		"""Creates random specifications of an image with multiple shapes.
		Returns the background color of the image and a list of shape
		specifications that can be used to draw the image.
		"""
		# Pick random background color
		bg_color = np.array([random.randint(0, 255) for _ in range(3)])
		#bg_color = np.array([0,0,0])

		# Generate a few random shapes and record their
		# bounding boxes
		shapes = []
		boxes = []
		N = random.randint(1, self.num_shapes)
		for _ in range(N):
			shape, color, dims, theta = self.random_shape(height, width)

			x, y, s, sg, ss, st = dims

			if shape == "stems":  # change here to account for orientation
				boxes.append([y - s, x - ss, y + s, x + ss])
				color = self.ColorS
			elif shape == "tabletop":
				boxes.append([y - s, x - st, y + s, x + st])
				color = self.ColorT
			elif shape == "rectangle" and (color in self.ColorT):
				boxes.append([y - s, x - sg, y + s, x + sg])
				#shape = 'tabletop'
			elif shape == "rectangle" and color in (self.ColorS):
				boxes.append([y - s, x - sg, y + s, x + sg])
				#shape = 'stems'
			elif shape == "rectangle":
				boxes.append([y - s, x - sg, y + s, x + sg])
			else:
				boxes.append([y - s, x - s, y + s, x + s])

			shapes.append((shape, color, dims, theta))
		# Apply non-max suppression wit 0.3 threshold to avoid
		# shapes covering each other
		keep_ixs = utils.non_max_suppression(
			np.array(boxes), np.arange(N), 0.3)
		shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
		return bg_color, shapes



def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



if __name__ == '__main__':
	import argparse

    # Parse command line arguments
	parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
	parser.add_argument("command",
	                    metavar="<command>",
	                    help="'train' or 'evaluate' on MS COCO")
	parser.add_argument('--dataset', required=False,
	                    metavar="/path/to/shapes/",
	                    help='Directory of the clusters dataset')
	parser.add_argument('--year', required=False,
	                    default=DEFAULT_DATASET_YEAR,
	                    metavar="<year>",
	                    help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
	parser.add_argument('--model', required=True,
	                    metavar="/path/to/weights.h5",
	                    help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
	                    default=DEFAULT_LOGS_DIR,
	                    metavar="/path/to/logs/",
	                    help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--limit', required=False,
	                    default=300,
	                    metavar="<image count>",
	                    help='Images to use for evaluation (default=300)')
	parser.add_argument('--download', required=False,
	                    default=False,
	                    metavar="<True|False>",
	                    help='Automatically download and unzip MS-COCO files (default=False)',
	                    type=bool)
	args = parser.parse_args()
	print("Command: ", args.command)
	print("Model: ", args.model)
	print("Dataset: ", args.dataset)
	print("Year: ", args.year)
	print("Logs: ", args.logs)
	print("Auto Download: ", args.download)


	# Configurations
	if args.command == "train":
	    config = ShapesConfig()
	else:
	    class InferenceConfig(ShapesConfig):
	        # Set batch size to 1 since we'll be running inference on
	        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	        GPU_COUNT = 1
	        IMAGES_PER_GPU = 1
	        DETECTION_MIN_CONFIDENCE = 0
	    config = InferenceConfig()
	config.display()



	############################################################
	#  Training
	############################################################

	# Training dataset
	dataset_train = ShapesDataset()
	dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
	dataset_train.prepare()

	# Validation dataset
	dataset_val = ShapesDataset()
	dataset_val.load_shapes(100 , config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
	dataset_val.prepare()



	# Load and display random samples
	image_ids = np.random.choice(dataset_train.image_ids, 5)
	for image_id in image_ids:
	    image = dataset_train.load_image(image_id)  # here it generates random images (4) instead, given the image_id, where it goes to see the specs of this id from the collection of random images_info (500) generated.  
	    mask, class_ids = dataset_train.load_mask(image_id)
	    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)  # we see 10 figures coming out sampling 4 images randomly selected among the 100 generated ones for teh training


	if args.command == "train":
		# Create model in training mode
		model = modellib.MaskRCNN(mode="training", config=config,
	                          model_dir=MODEL_DIR)

		# Which weights to start with?
		init_with = "coco"  # imagenet, coco, or last

		if init_with == "imagenet":
		    model.load_weights(model.get_imagenet_weights(), by_name=True)
		elif init_with == "coco":
		    # Load weights trained on MS COCO, but skip layers that
		    # are different due to the different number of classes
		    # See README for instructions to download the COCO weights
		    model.load_weights(COCO_MODEL_PATH, by_name=True,
		                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
		                                "mrcnn_bbox", "mrcnn_mask"]) # excluding some layers 
		elif init_with == "last":
		    # Load the last model you trained and continue training
		    model.load_weights(model.find_last(), by_name=True)



	class InferenceConfig(ShapesConfig):
	    GPU_COUNT = 1
	    IMAGES_PER_GPU = 1

	inference_config = InferenceConfig()

	# Recreate the model in inference mode
	model_inference = modellib.MaskRCNN(mode="inference", 
	                          config=inference_config,
	                          model_dir=MODEL_DIR)


	if args.command == "train":
		# Train the head branches
		# Passing layers="heads" freezes all layers except the head
		# layers. You can also pass a regular expression to select
		# which layers to train by name pattern
		mean_average_precision_callback_traindataset = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_train, calculate_map_at_every_X_epoch=10, verbose=1)


		model.train(dataset_train, dataset_val, 
		            learning_rate=config.LEARNING_RATE, 
		            epochs=5, #10, 2,10
		            layers='heads', custom_callbacks=[mean_average_precision_callback_traindataset])

		print('finished training on train dataset, head layers')

		# Fine tune all layers
		# Passing layers="all" trains all layers. You can also 
		# pass a regular expression to select which layers to
		# train by name pattern.
		model.train(dataset_train, dataset_val, 
		            learning_rate=config.LEARNING_RATE / 10,
		            epochs= 11, #21, 3, 50
		            layers="all", custom_callbacks=[mean_average_precision_callback_traindataset])

		print('finished training on train dataset, all layers')
		print('start getting last model path')


	# Save weights
	# Typically not needed because callbacks save after every epoch
	# Uncomment to save manually
	# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
	# model.keras_model.save_weights(model_path)





	# Get path to saved weights
	# Either set a specific path or find last trained weights
	# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
	model_path = model_inference.find_last()

	# Load trained weights
	print("Loading weights from ", model_path)
	model_inference.load_weights(model_path, by_name=True)



	# Test on a random imag
	for i in range(5):
		print('start inference on 5 random images form the val dataset')
		image_id = random.choice(dataset_val.image_ids)
		original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
		    modellib.load_image_gt(dataset_val, inference_config, 
		                           image_id, use_mini_mask=False)

		log("original_image_test_"+str(i), original_image)
		log("image_meta_test_"+str(i), image_meta)
		log("gt_class_id_test_"+str(i), gt_class_id)
		log("gt_bbox_test_"+str(i), gt_bbox)
		log("gt_mask_test_"+str(i), gt_mask)

		visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
		                            dataset_train.class_names, figsize=(8, 8))




		results = model_inference.detect([original_image], verbose=1) # verbose is for logging image

		r = results[0] # results is Returned as a list of dicts, one dict per image. dict contains: rois (=detection bbox) ...
		visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
		                            dataset_val.class_names, r['scores'], ax=get_ax())




	print('start loading simulation images to infer on ')
	Cimages = dataset_train.load_images_from_folder(CUSTOM_IMAGES_PATH)
	print('finish loading simulation images to infer on ')
	for cimg in Cimages:
		print('start testing on custom images')
		c_results = model_inference.detect([cimg], verbose=1)
		r = c_results[0]
		visualize.display_instances(cimg, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax()) 



	# Evaluation
	# Compute VOC-Style mAP @ IoU=0.5
	# Running on 10 images. Increase for better accuracy.
	image_ids = np.random.choice(dataset_val.image_ids, 100)
	APs = []
	for image_id in image_ids:
	    # Load image and ground truth data
	    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
	        modellib.load_image_gt(dataset_val, inference_config,
	                               image_id, use_mini_mask=False)
	    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0) #mold_image : substracts mean pixel from image and convert it to float, tey are molded maybe to ave small numbers of precisions (0-1 range)
	    # Run object detection
	    results = model_inference.detect([image], verbose=0)
	    r = results[0]
	    # Compute AP
	    AP, precisions, recalls, overlaps =\
	        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
	                         r["rois"], r["class_ids"], r["scores"], r['masks']) # rois is prediction bbox, AP = average precision among detections in 1 image
	    
	    ## Add Orientation extraction btanch
	    stems_orient = shapes_utils.get_orientation(results) 
	    APs.append(AP)
	    
	print("mAP from validation dataset : ", np.mean(APs))


## Run it by 1 of the following:
# python3 shapes_cluster.py command=train --model=coco
# python3 shapes_cluster.py command=train --model= ./../../mask_rcnn_coco.h5 


# refer to : https://github.com/matterport/Mask_RCNN/issues/1839
#for issue related to mAP onn training dataset


# best: /home/sariah/mask-rcnn/src/Mask_RCNN/logs/cluster_shapes20201216T0908/mask_rcnn_cluster_shapes_0020.h5
# epoch:21, iter_train:1000, it_val:300, mAP_train:0.95, mAP_val = 0.71