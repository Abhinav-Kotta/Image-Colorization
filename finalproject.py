# import numpy as np
# import tensorflow as tf
# import keras
# from keras import layers
# import cv2
# from keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout
# from keras.models import Sequential
# from keras.preprocessing.image import img_to_array
# import os
# from tqdm import tqdm
# import re
# import matplotlib.pyplot as plt
#
#
# # to get the files in proper order
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(data, key=alphanum_key)
#
#
# # defining the size of the image
# SIZE = 160
# color_img = []
# path = 'landscape Images/color'
# files = os.listdir(path)
# files = sorted_alphanumeric(files)
# for i in tqdm(files):
#     if i == '6000.jpg':
#         break
#     else:
#         img = cv2.imread(path + '/' + i, 1)
#         # open cv reads images in BGR format so we have to convert it to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # resizing image
#         img = cv2.resize(img, (SIZE, SIZE))
#         img = img.astype('float32') / 255.0
#         color_img.append(img_to_array(img))
#
# gray_img = []
# path = 'landscape Images/gray'
# files = os.listdir(path)
# files = sorted_alphanumeric(files)
# for i in tqdm(files):
#     if i == '6000.jpg':
#         break
#     else:
#         img = cv2.imread(path + '/' + i, 1)
#
#         # resizing image
#         img = cv2.resize(img, (SIZE, SIZE))
#         img = img.astype('float32') / 255.0
#         gray_img.append(img_to_array(img))
#
#
# # defining function to plot images pair
# def plot_images(color, grayscale):
#     plt.figure(figsize=(15, 15))
#     plt.subplot(1, 3, 1)
#     plt.title('Color Image', color='green', fontsize=20)
#     plt.imshow(color)
#     plt.subplot(1, 3, 2)
#     plt.title('Grayscale Image ', color='black', fontsize=20)
#     plt.imshow(grayscale)
#
#     plt.show()
#
#
#
#
# for i in range(3,10):
#      plot_images(color_img[i],gray_img[i])
#
# train_gray_image = gray_img[:5500]
# train_color_image = color_img[:5500]
#
# test_gray_image = gray_img[5500:]
# test_color_image = color_img[5500:]
# # reshaping
# train_g = np.reshape(train_gray_image,(len(train_gray_image),SIZE,SIZE,3))
# train_c = np.reshape(train_color_image, (len(train_color_image),SIZE,SIZE,3))
# print('Train color image shape:',train_c.shape)
#
#
# test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,3))
# test_color_image = np.reshape(test_color_image, (len(test_color_image),SIZE,SIZE,3))
# print('Test color image shape',test_color_image.shape)
#
# def down(filters , kernel_size, apply_batch_normalization = True):
#     downsample = tf.keras.models.Sequential()
#     downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
#     if apply_batch_normalization:
#         downsample.add(layers.BatchNormalization())
#     downsample.add(keras.layers.LeakyReLU())
#     return downsample
#
#
# def up(filters, kernel_size, dropout = False):
#     upsample = tf.keras.models.Sequential()
#     upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
#     if dropout:
#         upsample.dropout(0.2)
#     upsample.add(keras.layers.LeakyReLU())
#     return upsample
#
#
# def model():
#     inputs = layers.Input(shape=[160, 160, 3])
#     d1 = down(128, (3, 3), False)(inputs)
#     d2 = down(128, (3, 3), False)(d1)
#     d3 = down(256, (3, 3), True)(d2)
#     d4 = down(512, (3, 3), True)(d3)
#
#     d5 = down(512, (3, 3), True)(d4)
#     # upsampling
#     u1 = up(512, (3, 3), False)(d5)
#     u1 = layers.concatenate([u1, d4])
#     u2 = up(256, (3, 3), False)(u1)
#     u2 = layers.concatenate([u2, d3])
#     u3 = up(128, (3, 3), False)(u2)
#     u3 = layers.concatenate([u3, d2])
#     u4 = up(128, (3, 3), False)(u3)
#     u4 = layers.concatenate([u4, d1])
#     u5 = up(3, (3, 3), False)(u4)
#     u5 = layers.concatenate([u5, inputs])
#     output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
#     return tf.keras.Model(inputs=inputs, outputs=output)
#
#
# model = model()
# model.summary()
#
# print("Printed the summary. Moving onto compilation.")
#
# model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
#               loss = 'mean_absolute_error', metrics = ['acc'])
#
# print("Compiiled correctly.")
# model.fit(train_g, train_c, epochs = 50,batch_size = 50,verbose = 0)
# print("Model has been fit.")
# model.evaluate(test_gray_image,test_color_image)
# print("Model has been evaluated.")
#
# # defining function to plot images pair
# def plot_images(color, grayscale, predicted):
#     plt.figure(figsize=(15, 15))
#     plt.subplot(1, 3, 1)
#     plt.title('Color Image', color='green', fontsize=20)
#     plt.imshow(color)
#     plt.subplot(1, 3, 2)
#     plt.title('Grayscale Image ', color='black', fontsize=20)
#     plt.imshow(grayscale)
#     plt.subplot(1, 3, 3)
#     plt.title('Predicted Image ', color='Red', fontsize=20)
#     plt.imshow(predicted)
#
#     plt.show()
#
#
# for i in range(50, 58):
#     predicted = np.clip(
#         model.predict(test_gray_image[i].reshape(1, SIZE, SIZE, 3)), 0.0,
#         1.0).reshape(SIZE, SIZE, 3)
#     plot_images(test_color_image[i], test_gray_image[i], predicted)
#
#
#



'''Colorization autoencoder

The autoencoder is trained with grayscale images as input
and colored images as output.
Colorization autoencoder can be treated like the opposite
of denoising autoencoder. Instead of removing noise, colorization
adds noise (color) to the grayscale image.

Grayscale Images --> Colorization --> Color Images
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os

def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# load the CIFAR10 data
(x_train, _), (x_test, _) = cifar10.load_data()

# input image dimensions
# we assume data format "channels_last"
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

# create saved_images folder
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# display the 1st 100 input images (color and gray)
imgs = x_test[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

# convert color train and test images to gray
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# display grayscale version of test images
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()


# normalize output train and test color images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# normalize input train and test grayscale images
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

# reshape images to row x col x channel for CNN output/validation
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

# reshape images to row x col x channel for CNN input
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

# network parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256
# encoder/decoder number of CNN layers and filters per layer
layer_filters = [64, 128, 256]

# build the autoencoder model
# first build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
shape = K.int_shape(x)

# generate a latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# instantiate encoder model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# save weights for future use (e.g. reload parameters w/o training)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# called every epoch
callbacks = [lr_reducer, checkpoint]

# train the autoencoder
autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=30,
                batch_size=batch_size,
                callbacks=callbacks)

# predict the autoencoder output from test data
x_decoded = autoencoder.predict(x_test_gray)

# display the 1st 100 colorized images
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.show()