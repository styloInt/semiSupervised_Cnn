path_caffe = '/home/atemmar/caffe/';

import sys
from utils_dataSpine import *
import numpy as np
from PIL import Image

sys.path.insert(0, path_caffe + '/python')

import caffe

test_file = sys.argv[1]
rep_dataset = sys.argv[2]

model_deploy = sys.argv[3]
model_weights = sys.argv[4]
rep_save_results = sys.argv[5]

indices = open(test_file, 'r').read().splitlines()

files_x = [rep_dataset + line.split('\t')[0] for line in indices]
files_y = [rep_dataset + line.split('\t')[1] for line in indices]

# fileList = [f for f in list(os.walk(rep_traininSet))[0][2] if ".png" in f.lower()][:1000]
size_image = (96,304)
name_files = []
nb_image = 0
images_irm = np.zeros((size_image[0], size_image[1], len(files_x)))
labels = np.zeros((size_image[0], size_image[1], len(files_x)))
for file_x,file_y in zip(files_x, files_y):
    im = np.array(Image.open(file_x))
    im = preprocessing_im(im)
    label = np.array(Image.open(file_y))
    label = preprocessing_label(label) #resize

    images_irm[:,:,nb_image] = im
    labels[:,:,nb_image] = label
    
    nameFile = "/".join(file_x[:-4].split("/")[-2:])
    name_files.append(nameFile)
    
    nb_image += 1

caffe.set_device(0)
caffe.set_mode_cpu()
# solver = caffe.SGDSolver("/home/atemmar/Documents/Data/FromFirasPng/models/U-net/solver_unet_softmax.prototxt")
# solver.restore("/home/atemmar//Documents/Data/FromFirasPng/models_pretrained/U-net/train_unet_spine_softmax_iter_40000.solverstate")
# net_deploy=solver.net
net_deploy = caffe.Net(model_deploy,      # defines the structure of the model
                model_weights, caffe.TEST)


save_results(dataset=images_irm, labels=labels, net_deploy=net_deploy, rep_save_results=rep_save_results, nameFiles=name_files)





