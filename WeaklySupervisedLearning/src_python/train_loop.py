path_caffe = '/home/atemmar/caffe/';
# path_caffe = "/home/luffy/Documents/Stage/"

import sys
from utils_dataSpine import *
import numpy as np
from PIL import Image
from shutil import copyfile
from skimage import data, segmentation, color
from skimage.future import graph
sys.path.insert(0, path_caffe + '/python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



rep_dataset = "/home/atemmar/Documents/Data/FromFirasPng/"
#rep_dataset = "/home/luffy/Documents/Stage/semiSupervised_Cnn/FromFirasPng/"

orginal_net = rep_dataset + "/models/U-net/unet.prototxt"
solver_name = rep_dataset + "/models/U-net/solver_unet_softmax_iter.prototxt"
k = int(sys.argv[1])
lamda = float(sys.argv[2])
original_train_filenames = rep_dataset + "train.txt"
unlabelled_files_name = rep_dataset + "unlab.txt"
test_files_name = rep_dataset + "test.txt"
number_epoch = 20
repSavePrediction = rep_dataset
nameTmpTrainFiles = rep_dataset + "/train_tmp.txt"
weight_file = rep_dataset + "/models_pretrained/U-net/train_unet_spine_softmax_iter_460000.caffemodel"
net_deploy_name = rep_dataset + "/models/U-net/unet_deploy.prototxt"
dice_saveUnlab_filename = rep_dataset + "/dicesResuls_unlab_" + str(k) + "_" + str(lamda) +".txt"
normalize = "normalizeCut"

epoch_start = 0

prefix_snapshot = rep_dataset + "/models_pretrained/U-net_iter/train_unet_spine_softmax_" + str(k) + "_" + str(lamda) +"_iter";
	

def replace_labelFiles(prototxt, old_labelName, newLabelFileName):

	# Step 1 : read file and save the content
	train_prototxt = open(prototxt, "r")
	train_prototxt_lines = train_prototxt.readlines()

	# Step 2: rewrite it and replace the labelFileName
	train_prototxt = open(prototxt, "w")

	for line in train_prototxt_lines:
		if old_labelName in line:
			line = line.replace(old_labelName, newLabelFileName)

		train_prototxt.write(line)

	train_prototxt.close()

def getPrototxtFromSolver(solver_name):

	solver_file = open(solver_name, "r")

	lines = solver_file.readlines()

	for line in lines:
		split = line.split(":")
		if 'net' in split [0]: # if this line corresponds to the line give the network file
			solver_file.close()
			return split[1].replace("\"", "").replace(" ", "")[:-1] # we remove the "" and th \n

	solver_file.close()
	return None

def modifySnapshotPrefixSolver(solver_name, new_prefix):
	solver_file = open(solver_name, "r")
	lines = solver_file.readlines()

	newLines = []
	for line in lines:
		split = line.split(":")
		if 'snapshot_prefix' in split [0]: # if this line corresponds to the line give the network file
			split[1] = "\"" + new_prefix + "\""
			newLines.append(split[0] + " : " + split[1] + '\n')
		else:
			newLines.append(line)

	solver_file.close()

	#we rewrite the file
	solver_file = open(solver_name, "w")

	for line in newLines:
		solver_file.write(line)

	solver_file.close()


def normalize_cut(img):
	if img.shape[2] < 3:
		img *= 255
		img = img.astype(np.uint8)
		img = np.append(img, np.zeros((img.shape[0], img.shape[1],1), dtype=np.uint8), 2)

	labels1 = segmentation.slic(img, compactness=30, n_segments=400)
	g = graph.rag_mean_color(img, labels1, mode='similarity')
	labels2 = graph.cut_normalized(labels1, g)
	out2 = color.label2rgb(labels2, img, kind='avg')

	return out2.argmax(2)



# #################### INITIALISATION ########################### #

print ("Initialisation ...")

images_unlab, labels_unlab = load_dataset(unlabelled_files_name, rep_dataset) # we are not suppose to have the labels of course, it's for mesuring the improvement
images_test, labels_test = load_dataset(test_files_name, rep_dataset)

# load the unlabelled files names
unlabelled_files_name_file = open(unlabelled_files_name, "r")
unlabelled_inputs = []
unlabelled_original_output = []
for line in unlabelled_files_name_file.readlines():
	split = line.split("\t")
	unlabelled_inputs.append(split[0])
	unlabelled_original_output.append(split[1][:-1])

unlabelled_files_name_file.close()


if epoch_start != 0:
	weight_file = prefix_snapshot + "_iter_" + str((epoch_start) * k) + ".solverstate"
else:
	copyfile(original_train_filenames, nameTmpTrainFiles)

dice_saveUnlab_file = open(dice_saveUnlab_filename, "a")


# Copy the original file in a tmp file
network_prototxt = getPrototxtFromSolver(solver_name)

#modify the snapshot prefix to fit the parameters
modifySnapshotPrefixSolver(solver_name, prefix_snapshot)

copyfile(orginal_net, network_prototxt)
replace_labelFiles (network_prototxt, original_train_filenames, nameTmpTrainFiles)

solver = caffe.SGDSolver(solver_name)
if "solverstate" in weight_file:
	solver.restore(weight_file)
else:
	solver.net.copy_from(weight_file)

print ("end of initialisation")


# #################### TRAINING ########################### # 
print ("training...")
for num_epoch in range(epoch_start, number_epoch):
	# caffe.set_mode_gpu()

	# Run the algorithm for k itterations, give a weight lamdda for the unlabelled data
	solver.step(k)
	solver = None
	# #################### PREDICTION ON UNLABALLED  data ########################### # 
	# Use graph cut or trust region to improve the segmentation. save the results
	print ("Prediction ...")
	new_weight_file = prefix_snapshot + "_iter_" + str((num_epoch+1) * k) + ".caffemodel"
	net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
	                new_weight_file, caffe.TEST)

	copyfile(original_train_filenames, nameTmpTrainFiles)
	train_tmp_file = open(nameTmpTrainFiles, "a")
	predictions = np.zeros(labels_unlab.shape)
	for num_image in range(images_unlab.shape[2]):
		img = images_unlab[:,:, num_image]
		heatmap = get_heat_map(img, net_deploy)

		if normalize == "normalizeCut":
			predictions[:,:,num_image] = normalize_cut(heatmap)
		else:
			predictions[:,:,num_image] = heatmap.argmax(2)

		nameFile = "./prediction_tmp/" + str(k) + "_" + str(lamda) + "/epoch_" + str(num_epoch) + "/" + "/".join(unlabelled_original_output[num_image].split("/")[-2:])
		nameFileWeight = "./prediction_tmp/" + str(k) + "_" + str(lamda) + "/epoch_" + str(num_epoch) + "/" + "/".join((unlabelled_original_output[num_image][:-4] + "_weight.png").split("/")[-2:])

		createDirectoryPath(repSavePrediction + nameFile)
		scipy.misc.toimage(predictions[:,:,num_image], cmin=0, cmax=255).save(repSavePrediction + nameFile)

		weight = np.zeros((img.shape)) + lamda
		scipy.misc.toimage(weight, cmin=0, cmax=1).save(repSavePrediction + nameFileWeight)

		train_tmp_file.write(unlabelled_inputs[num_image] + "\t" + nameFile + "\t" + nameFileWeight +"\n")

	print ("End of prediction ")
	net_deploy = None
	train_tmp_file.close()

	print("Compute dice on the unlabel files .... ")
	dice = np.mean(compute_dice_dataset(labels_unlab, predictions))
	print ("Dice : ", dice)
	dice_saveUnlab_file.write(str(num_epoch) + " : " + str(dice) + "\n")
	dice_saveUnlab_file.close()
	dice_saveUnlab_file = open(dice_saveUnlab_filename, "a")

	solver = caffe.SGDSolver(solver_name)
	solver.restore(new_weight_file.replace("caffemodel", 'solverstate'))

print ("end of training")

