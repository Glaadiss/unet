from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from sklearn.model_selection import train_test_split

Street = [0, 0, 255]
Building = [255, 0, 0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Street, Building, Unlabelled])


im_width = 512
im_height = 512
border = 1
imagesNumber = 100
path_image = './data/berlin/image/'
path_label = './data/berlin/label/'


def get_path(path_type, id):
    return path_type + str(id) + '.png'



def get_data(): 
    X = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((imagesNumber, im_height, im_width, 1), dtype=np.float32)

    for id in range(1, imagesNumber + 1):        

        x_img = img_to_array(load_img(get_path(path_image, id), color_mode="grayscale" ))        
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)

        mask_org = img_to_array(load_img(get_path(path_label, id)))
        mask_org[np.where((mask_org==[255,0, 0]).all(axis=2))] = [255,255,255]
        mask = rgb2gray(mask_org)
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)                

        X[id-1] = x_img / 255
        y[id-1] = mask / 255

    return X, y




def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out



def saveResult(save_path, npyfile, originals, ground_truths, flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        # img = labelVisualize(num_class,COLOR_DICT,item)   
        # print(np.unique(img))      
        img = item[:,:,0] if len(item.shape) == 3 else item
        img = img_as_ubyte(img)

        original = originals[i][:, :, 0] if len(originals[i].shape) == 3 else originals[i]
        original = img_as_ubyte(original)

        ground_truth = ground_truths[i][:, :, 0] if len(ground_truths[i].shape) == 3 else ground_truths[i]
        ground_truth = img_as_ubyte(ground_truth)
        

        merge = np.concatenate((img, original, ground_truth), axis=1 )
        io.imsave(os.path.join(save_path,"%d_predict.png"%i), merge)