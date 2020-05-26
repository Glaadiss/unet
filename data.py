from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

Street = [0, 0, 255]
Building = [255, 0, 0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Street, Building, Unlabelled])


im_width = 512
im_height = 512
border = 1
imagesNumber = 2
path_image = './data/berlin/image/'
path_label = './data/berlin/label/'


def get_path(path_type, id):
    return path_type + str(id) + '.png'



def get_data(): 
    X = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)
    orig_X = np.zeros((imagesNumber, im_height, im_width, 3), dtype=np.float32)
    orig_y = np.zeros((imagesNumber, im_height, im_width, 3), dtype=np.float32)

    for id in range(1, imagesNumber + 1):        

        x_img_org = img_to_array(load_img(get_path(path_image, id)))        
        x_img_org = resize(x_img_org, (im_height, im_width, 3), mode='constant', preserve_range=True)
        x_img = rgb2gray(x_img_org)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        mask_org = img_to_array(load_img(get_path(path_label, id)))
        mask_org[np.where((mask_org==[255,0, 0]).all(axis=2))] = [255,255,255]
        mask_org = resize(mask_org, (im_height, im_width, 3), mode='constant', preserve_range=True)                
        mask = rgb2gray(mask_org)
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)                

        X[id-1] = x_img / 255
        y[id-1] = mask / 255
        orig_X[id-1] = x_img_org
        orig_y[id-1] = mask_org

    return X, y, orig_X, orig_y




def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out 



def saveResult(save_path, npyfile, originals, ground_truth, flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item)         
        merge = np.concatenate((img, originals[i], ground_truth[i]), axis=1 )
        io.imsave(os.path.join(save_path,"%d_predict.png"%i), merge)