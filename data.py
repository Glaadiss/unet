from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


Street = [0, 0, 255]
Building = [255, 0, 0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Street, Building, Unlabelled])


im_width = 256
im_height = 256
border = 5
imagesNumber = 4
path_image = './data/berlin/image/'
path_label = './data/berlin/label/'


def get_path(path_type, id):
    return path_type + str(id) + '.png'


def get_data(): 
    X = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)

    for id in range(1, imagesNumber + 1):        
        img = load_img(get_path(path_image, id), grayscale=True)
        x_img = img_to_array(img)        
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        mask = img_to_array(load_img(get_path(path_label, id), grayscale=True))
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
        X[id-1] = x_img / 255
        y[id-1] = mask / 255

    return X, y




def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)