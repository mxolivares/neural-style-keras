'''
Fast Style Transfer using a webcam

'''

from __future__ import print_function
import os
import argparse

import numpy as np
import tensorflow as tf
import keras

import cv2
import h5py
import yaml
import time

import keras.backend as K

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from scipy.misc import imsave
from PIL import Image

from model import pastiche_model
from utils import config_gpu, preprocess_image_scale, deprocess_image


def preprocess_webcam_capture(cam_image, img_size=None):

    img = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if img_size:
        scale = float(img_size) / max(img.size)
        new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
        img = img.resize(new_size, resample=Image.BILINEAR)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use a trained pastiche network.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
    parser.add_argument('--img_size', type=int, default=720) #1024)
    parser.add_argument('--use_style_name', default=False, action='store_true')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)
    checkpoint_path = os.path.splitext(args.checkpoint_path)[0]

    with h5py.File(checkpoint_path + '.h5', 'r') as f:
        model_args = yaml.load(f.attrs['args'])
        style_names = f.attrs['style_names']

    print('Creating pastiche model...')
    class_targets = K.placeholder(shape=(None,), dtype=tf.int32)
    pastiche_net = pastiche_model(None, width_factor=model_args.width_factor,
                                  nb_classes=model_args.nb_classes,
                                  targets=class_targets)
    with h5py.File(checkpoint_path + '.h5', 'r') as f:
        pastiche_net.load_weights_from_hdf5_group(f['model_weights'])

    inputs = [pastiche_net.input, class_targets, K.learning_phase()]

    transfer_style = K.function(inputs, [pastiche_net.output])

    cam = cv2.VideoCapture(0)

    while True:

        _, cam_image = cam.read()

        img = preprocess_webcam_capture(cam_image, img_size=args.img_size)

        style_name = style_names[0]
        indices = [1]
        out = transfer_style([img, indices, 0.])[0]

        result = deprocess_image(out[0][None, :, :, :].copy())

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('result', result)
        if cv2.waitKey(100) == 27:
            cv2.destroyAllWindows()
            break
