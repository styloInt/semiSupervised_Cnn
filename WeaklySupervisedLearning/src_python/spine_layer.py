import caffe

import numpy as np
from PIL import Image
import scipy.io

import random
import cv2

from skimage import img_as_uint

import numpy as np

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

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

class Spine_layer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL-Context
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 59 class task defined by

        R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R.
        Urtasun, and A. Yuille.  The Role of Context for Object Detection and
        Semantic Segmentation in the Wild.  CVPR 2014.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - rv_dir: path to RV dataset
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for RV semantic segmentation.

        example: params = dict(voc_dir="/path/to/PASCAL", split="val")
        """
        # config
        params = eval(self.param_str)
        self.dir = params['dir']
        # Image template for applying histogram transfer
        self.apply_ht = False

        self.inputLabelFiles = params['inputLabelFiles']


        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        indices = open(self.inputLabelFiles, 'r').read().splitlines()

        # load indices for images and labels
        self.indices_x = [line.split('\t')[0] for line in indices]
        self.indices_y = [line.split('\t')[1] for line in indices]


        new_shape = (96,304)

        self.idx = 0
        self.inputs = np.zeros((new_shape[0], new_shape[1], len(self.indices_x)))
        self.labels = np.zeros((new_shape[0], new_shape[1], len(self.indices_x)))

        self.nb_images = 0
        for i in range(len(self.indices_x)):
            label = self.load_label(self.indices_y[i])
            if (len(np.unique(label)) == 2) :
                data = self.load_image(self.indices_x[i])
                # self.inputs[:,:,self.nb_images] = cv2.resize(data, (new_shape[1], new_shape[0]))
                # self.labels[:,:,self.nb_images] = cv2.resize(label, (new_shape[1], new_shape[0]))
                self.inputs[:,:,self.nb_images] = data
                self.labels[:,:,self.nb_images] = label
                self.nb_images += 1

        self.weights = np.zeros((new_shape[0], new_shape[1], len(self.indices_x)))
        default_weight = np.ones(new_shape)
        for i, line in enumerate(indices): 
            split_line = line.split('\t')
            if len(split_line) > 2: #weight are defined
                self.weights[:,:,i] = im2double(np.array(Image.open('{}/{}'.format(self.dir, split_line[2]))))
            else:
                self.weights[:,:,i] = default_weight

        # make eval deterministic
        # if 'train' not in self.split:
        #     self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, self.nb_images)


        if "image_template" in params:
            im = np.array(Image.open(params['image_template']))
            if (len(im.shape) > 2):
                im = im[:,:,0]
            self.image_template = im
            self.apply_ht = True

        #read mean_image
        # mean_image_path = params['mean_image']
        # self.mean_image = Image.open(mean_image_path)

    def reshape(self, bottom, top):
        # load image + label image pair
        # self.data = self.load_image(self.indices_x[self.idx])
        # self.label = self.load_label(self.indices_y[self.idx])

        # new_shape = (304,96)
        # self.data = cv2.resize(self.data, (new_shape[0], new_shape[1]))
        # self.label = cv2.resize(self.label, (new_shape[0], new_shape[1]))

        self.data = self.inputs[:,:,self.idx]
        self.label = self.labels[:,:, self.idx]
        self.weight = self.weights[:,:, self.idx]

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, 1,*self.data.shape)
        top[1].reshape(1, 1,*self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

        # pick next input
        if self.random:
            self.idx = random.randint(0, self.nb_images)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(Image.open('{}/{}'.format(self.dir, idx)))
        if (len(in_.shape) > 2):
            in_ = in_[:,:,0]

        if self.apply_ht: #apply transfer histogram
            in_ = hist_match(in_, self.image_template)
        in_ = np.array(in_, dtype=np.float32)

        # in_ = in_[:,:,::-1]
        #in_ -= self.mean_image
        max_img, min_img = np.max(in_), np.min(in_)
        in_ = 2*(in_-max_img)/(max_img - min_img) - 1

        in_ = in_[107:203,16:]
        # in_ = in_
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        # print idx
        label = np.array(Image.open('{}/{}'.format(self.dir, idx)))

        # if multiple channel, transform in one channel, with three distinct channel

        if (len(label.shape) > 2):
            label = label[:,:,0]
            label[label == 255] = 1

        # print "Unique : ", np.unique(label)

        label = label[107:203,16:]
        label = label.astype(np.uint8)
        # label = label[np.newaxis, ...]
        return label
