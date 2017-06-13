path_caffe = '/home/atemmar/caffe/';

import sys
from utils_dataSpine import *
import numpy as np
from PIL import Image

sys.path.insert(0, path_caffe + '/python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

solver_file = sys.argv[1]
k = int(sys.argv[2])
unlabelled_files_name = sys.argv[3]
test_files_name = sys.argv[4]
rep_dataset = sys.argv[5]

# #################### INITIALISATION ########################### #

images_unlab, labels_unlab = load_dataset(unlabelled_files_name, rep_dataset) # we are not suppose to have the labels of course, it's for mesuring the improvement
images_test, labels_test = load_dataset(test_files_name, rep_dataset)

solver = caffe.SGDSolver(solver_file)

# weight_file = None
# if len(sys.argv) > 3:
# 	weight_file = sys.argv[3]
	# solver.net.copy_from(weight_file)

solver_file = None
if len(sys.argv) > 3:
	solver_state = sys.argv[3]
	solver.restore(solver_state)


# #################### TRAINING ########################### # 
solver.step(k)


# #################### PREDICTION ON UNLABALLED  data ########################### # 
# Use graph cut or trust region to improve the segmentation. save the results
net_deploy = caffe.Net(sys.argv[4],      # defines the structure of the model
                model_weights, caffe.TEST)

for num_image in images_unlab.shape[2]:
	img = images_unlab[:,:, num_image]
	heatmap = get_heat_map(img, net_deploy)

    
    nameFile = "/".join(file_x[:-4].split("/")[-2:])
    name_files.append(nameFile)
    
    nb_image += 1


# Run the algorithm for k itterations, give a weight lamdda for the unlabelled data

# Compute the dice results for image gt, unlabelled image and test set. save the results 

# loop

