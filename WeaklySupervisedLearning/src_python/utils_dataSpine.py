import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage
# import cv2
import time
import scipy.misc
import caffe
import seaborn as sns
from IPython import display
import time 
from PIL import Image
import random
import cv2
import scipy.io

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


#devide an image in four pieces
def devide_four(img, heigth_block, width_block):
    top_left = img[:heigth_block,:width_block,:]
    top_right = img[:heigth_block,width_block:,:]
    bot_left = img[heigth_block:,:width_block,:]
    bot_right = img[heigth_block:,width_block:,:]

    return top_left, top_right, bot_left, bot_right

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize_heatmap(heatmap):
    heat_map_normalize = np.zeros(heatmap.shape)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heat_map_normalize[x,y,:] = softmax(heatmap[x,y,:])
    
    return heat_map_normalize

def do_training(solver, step_size, nb_step=0):
        solver.step(step_size)

        heat_map = solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(heat_map)
#         heat_map_normalize = heat_map
        minimum = np.min(heat_map[:,:,0])

        nb_subplot = 4
        plt.figure(figsize=(10,10))
        image_test = solver.test_nets[0].blobs["data"].data[0,0,:,:]
        image_test_label = solver.test_nets[0].blobs["label"].data[0,0,:,:]
        plt.subplot(1,nb_subplot,1)
        plt.imshow(image_test)
        plt.title("image test")
        plt.subplot(1,nb_subplot,2)
        plt.imshow(image_test_label)
        plt.title("Label of the test image")
        plt.subplot(1,nb_subplot,3)
        plt.imshow(np.append(heat_map_normalize, np.zeros((heat_map_normalize.shape[0], heat_map_normalize.shape[1],1)), 2))
        plt.title("Heat map")
        # plt.subplot(1,nb_subplot,4)
        # plt.imshow(np.append(heat_map_normalize, np.zeros(heat_map_normalize.shape[0], heat_map_normalize.shape[1],1), 3))
        # plt.title("score")
        plt.subplot(1,nb_subplot,4)
        plt.imshow(solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0).argmax(2), vmin=0, vmax=1)
        plt.title("After : " + str(nb_step+step_size) + " itterations")
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(1)
    
###
	# save_image : place where to save the image
###
def save_image(img, vmin=None, vmax=None, title='', save_image=None, save_asMat=False):
	# plt.imshow(img, vmin=vmin, vmax=vmax)
	# plt.axis('off')
	# plt.title(title)

    if not save_image is None: #if not nans
        path = os.path.dirname(save_image)
        if not os.path.exists(path):
            os.makedirs(path)

    scipy.misc.toimage(img, cmin=vmin, cmax=vmax).save(save_image)

    if save_asMat:
        scipy.io.savemat(save_image[:-4] + ".mat", mdict={os.path.basename(save_image)[:-4] : img})

    # figure = plt.figure(figsize=(img.shape[0], img.shape[1]), dpi=1)
    # axis = plt.subplot(1, 1, 1)
    # plt.axis('off')
    # plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    # plt.imshow(img, vmin=vmin, vmax=vmax)


    # extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    # plt.savefig(save_image, format='jpeg', bbox_inches=extent, pad_inches=0)
    # plt.close(figure)

    # image = Image.fromarray(img * 255/(255-np.))
    # image.save(save_image)


def dice_metric(seg, gt):
    if (float(np.count_nonzero(seg[seg!=0]) + np.count_nonzero(gt[gt!= 0])) == 0):
        return 1

    if(np.count_nonzero(seg[gt!=0]) == 0 and np.count_nonzero(gt) == 0):
        return 1


    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    # print dice
    return dice

def preprocessing_im(im):
    newSize = (96, 304)
    if im.shape != newSize:
        if len(im.shape) > 2:
            im = im[:,:,0]

        im = im.astype(np.float)

        max_img, min_img = np.max(im), np.min(im)
        im = 2*(im-max_img)/(max_img - min_img) - 1

        im = im[110:200,:]
        im = cv2.resize(im, (newSize[1], newSize[0]))

    return im

def preprocessing_label(label):
    newSize = (96, 304)
    if label.shape != newSize:
        if (len(label.shape) > 2):
            label = label[:,:,0]
            label[label == 255] = 1

        label = label[110:200,:]
        label = cv2.resize(label, (newSize[1], newSize[0]))

    return label


def compute_recall(label, label_predicted):
    right_prediction = (label == label_predicted).astype(int)
    nb_pos_predict = nb.sum(right_prediction[label != 0])
    nb_pos = np.sum((label != 0).astype(int))
    
    return float(nb_pos_predict/nb_pos)


def compute_dice_dataset(dataset, gts, net_deploy):
    dices = []
    for num_image in range(dataset.shape[2]):
        img = dataset[:,:,num_image]
        img = preprocessing_im(img)

        label = gts[:,:, num_image]
        label = preprocessing_label(label)

        net_deploy.blobs['data'].data[...] = img
        net_deploy.forward()
        out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(out)
        label_out = heat_map_normalize.argmax(2)

        dices.append(dice_metric(label_out, label))

    return dices

def compute_dice_dataset(gts, prediction):
    dices = []
    for num_image in range(gts.shape[2]):
        dices.append(dice_metric(gt=gts[:,:,num_image], seg=prediction[:,:,num_image]))

    return dices


def get_heat_map(img, net_deploy):
    img = preprocessing_im(img)
    net_deploy.blobs['data'].data[...] = img
    net_deploy.forward()
    out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
    return normalize_heatmap(out)

def get_prediction(img, net_deploy):
    heatmap = get_heat_map(img, net_deploy)
    return heatmap.argmax(axis=2)

def get_predictions(imgs, net_deploy):
    predictions = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    for num_image in range(imgs.shape[2]):
        predictions[:,:,num_image] = get_prediction(imgs[:,:,num_image], net_deploy)

    return predictions


def load_dataset(file_names, rep_dataset, readInputs=True, readGT=True):
    size_image = (96,304)

    file_names_file = open(file_names, "r")
    lines = file_names_file.readlines()
    files_x = [rep_dataset + line.split('\t')[0] for line in lines]
    files_y = [rep_dataset + line.split('\t')[1][:-1] for line in lines]
    file_names_file.close()

    images = np.zeros((size_image[0], size_image[1], len(files_x)))
    labels = np.zeros((size_image[0], size_image[1], len(files_x)))

    nb_image = 0
    for file_x, file_y in zip(files_x, files_y):
        if readInputs:
            im = np.array(Image.open(file_x))
            images[:,:,nb_image] = preprocessing_im(im)

        if readGT:
            label = np.array(Image.open(file_y))
            labels[:,:,nb_image] = preprocessing_label(label)

        nb_image += 1


    if readInputs and readGT:
        return images, labels
    elif readInputs:
        return images
    elif readGT:
        return labels
    elif readInputs:
        return images



"""
	nbImageToSave : if the dataset is too big, you can choose one 
"""
def save_results(dataset,labels, net_deploy, rep_save_results = None, nbImageToSave = None, nameFiles=None):
    indices = list(range(dataset.shape[-1]))
    images_to_test = indices[:nbImageToSave]

    nbImageToDisplay = 4
    has_gt = True
    if labels is None :
        labels = np.zeros(dataset.shape)
        nbImageToDisplay = 3
        has_gt = False


    for num_image in images_to_test:
        img = dataset[:,:,num_image]
        img = preprocessing_im(img)

        label = labels[:,:, num_image]
        label = preprocessing_label(label)


        net_deploy.blobs['data'].data[...] = img
        net_deploy.forward()
        out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(out)
        label_out = heat_map_normalize.argmax(2)

        if rep_save_results is None:
        	name_save_image = [None] * 4
        elif not nameFiles is None:
            name_save_image = [rep_save_results + nameFiles[num_image] + "_ori.jpg", rep_save_results + nameFiles[num_image] + "_gt.jpg", \
            rep_save_results + nameFiles[num_image] + "_hm.jpg", rep_save_results + nameFiles[num_image] + "_predict.jpg"]
        else:
        	name_save_image = [rep_save_results + str(num_image) + "_ori.jpg", rep_save_results + str(num_image) + "_gt.jpg", \
        	rep_save_results + str(num_image) + "_hm.jpg", rep_save_results + str(num_image) + "_predict.jpg"]

        index_plot = 1

        # plt.figure(figsize=(10,10))
        # plt.subplot(1,nbImageToDisplay,index_plot); index_plot += 1
        save_image(img, title="Orginal image", save_image=name_save_image[0])
        if has_gt:
            # plt.subplot(1,nbImageToDisplay,index_plot); index_plot += 1
            save_image(label, vmin=0, vmax=1, title="Ground truth segmentation", save_image=name_save_image[1])
        # plt.subplot(1,nbImageToDisplay,index_plot); index_plot += 1
        save_image(np.append(heat_map_normalize, np.zeros((heat_map_normalize.shape[0], heat_map_normalize.shape[1],1)), 2), vmin=0, vmax=1, title="Heat map", save_image=name_save_image[2])
        # plt.subplot(1,nbImageToDisplay,index_plot); index_plot += 1
        save_image(label_out, vmin=0, vmax=1, title="Segmentation predicted", save_image=name_save_image[3])


def createDirectoryPath(file_name):
    path = os.path.dirname(file_name)
    if not os.path.exists(path):
        os.makedirs(path)

