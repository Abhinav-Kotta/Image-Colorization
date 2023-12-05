import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras import backend as K
from keras.models import load_model

'---------------'
'PREPARING THE DATASETS'
'---------------'
# Load in the images from the cifar10 dataset.
(x_train, _), (x_test, _) = cifar10.load_data()

# Turn the images in the test dataset to grayscale for a new test dataset
x_train_gray = np.dot(x_train[...,:3], [0.299, 0.587, 0.114])
x_test_gray = np.dot(x_test[...,:3], [0.299, 0.587, 0.114])

# Get the size of the images
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

'---------------'
'PLOTTING THE DATASETS'
'---------------'
# Store 100 images of the color test dataset
datasetC = x_test[:100]
datasetC = datasetC.reshape((10, 10, img_rows, img_cols, channels))
datasetC = np.vstack([np.hstack(i) for i in datasetC])

# Plot 100 images of the color test dataset
plt.figure()
plt.axis('off')
plt.title('Test Images (Color)')
plt.imshow(datasetC, interpolation='none')
plt.show()

# Store 100 images of the grayscale test dataset
datasetG = x_test_gray[:100]
datasetG = datasetG.reshape((10, 10, img_rows, img_cols))
datasetG = np.vstack([np.hstack(i) for i in datasetG])

# Plot 100 images of the grayscale test dataset
plt.figure()
plt.axis('off')
plt.title('Test Images (Grayscale)')
plt.imshow(datasetG, interpolation='none', cmap='gray')
plt.show()

'---------------'
'PREPARING FOR MODEL INPUT/OUTPUT'
'---------------'
# Normalize and reshape the color datasets
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

# Normalize and reshape the grayscale datasets
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

# Define parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256
layer_filters = [64, 128, 256]

'---------------'
'BUILDING THE ENCODER MODEL'
'---------------'

# Build the model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

# 3 Convolution layers: 64 filters, 128 filters, 256 filters
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Reshape to 4x4x256 for processing, to eventually be shaped back into 32x32x3
shape = K.int_shape(x)

# Thank you for resources online on how to do this type of thing.
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

encoder = Model(inputs, latent, name='encoder')
encoder.summary()

'---------------'
'BUILDING THE DECODER MODEL'
'---------------'
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Three convolution layers: 256 filters, 128 filters, 64 filters
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

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

'---------------'
'THE AUTOENCODER'
'---------------'
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# Save as a new model any time the loss improves.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Model{epoch:02d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# Save weights
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# Called every epoch
callbacks = [lr_reducer, checkpoint]

# Train the autoencoder
autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=30,
                batch_size=batch_size,
                callbacks=callbacks)

# Predict the autoencoder output from test data
x_decoded = autoencoder.predict(x_test_gray)

# Display the 1st 100 colorized images
datasetGtoC = x_decoded[:100]
datasetGtoC = datasetGtoC.reshape((10, 10, img_rows, img_cols, channels))
datasetGtoC = np.vstack([np.hstack(i) for i in datasetGtoC])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(datasetGtoC, interpolation='none')
plt.show()
