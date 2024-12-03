from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io as scio
import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

import os


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist",save_dir='result/2401/3000/128_50_2_CMA/1', filename = 'vae_mean.png'):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)



    filename = 'vae_mean.png'
    # display a 2D plot of the FC classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)#
    plt.colorbar()
    plt.xlabel("Z0")
    plt.ylabel("Z1")
    #plt.xlim(-6, 6)  # -6~6
    #plt.ylim(-6, 6)
    plt.savefig(save_dir+'/'+filename)
    plt.show()

    filename = 'FCs_over_latent.png'
    # display a 30x30 2D manifold of FCs
    n = 10
    # digit_size = 90
    digit_size = 68
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of FC classes in the latent space
    grid_x = np.linspace(-6, 6, n)
    grid_y = np.linspace(-6, 6, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("Z0")
    plt.ylabel("Z1")
    plt.imshow(figure)  # cmap='Greys_r')
    plt.grid(color='w', linewidth=2)
    plt.savefig(save_dir+'/'+filename)
    plt.show()


# MNIST dataset
# load data
data = scio.loadmat('data/2401/3000/34fc_new/2/FC_3000_CMA.mat')
mdata = np.array(data['FC_9000'])
#mdata2 = np.array(list(np.array(mdata).flatten()))

my_data = mdata
la=scio.loadmat('data/2401/3000/34fc_new/2/label_3000_CMA.mat')
label = np.array(la['label_9000'])

# my_data1 = np.squeeze(my_data[np.where((label==1)|(label==2)),:])
# label1 = label[np.where((label==1)|(label==2))]
my_data1 = my_data


# split data in train and test (the data is randomized before )
x_train = my_data1[0:int(len(my_data1) * 0.8)]
x_test = my_data1[int(len(my_data1) * 0.8) + 1:len(my_data1)]

label_1 = label
y_train = label_1[0:int(len(my_data1) * 0.8)]
y_test = label_1[int(len(my_data1) * 0.8) + 1:len(my_data1)]
#auto split data in train , test, validation
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(my_data1, label_1, test_size = 0.20, random_state = 0)
# x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0)
# print(len(x_train),len(x_valid),len(x_test)) 

# original_dim = 11781
original_dim = 4624
# original_dim = 23716
#original_dim = 68

# network parameters
input_shape = (original_dim,)
intermediate_dim = 1028
# intermediate_dim = 5800
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')#
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
data = (x_test, y_test)

#loss function
# x = K.flatten(x)
# z_decoded = K.flatten(z_decoded)
# reconstruction_loss = original_dim*keras.metrics.binary_crossentropy(K.flatten(inputs), K.flatten(outputs)) 
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim

# kl_loss = -5e-1 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1 ) 
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

#earlyStop
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min', verbose=1, restore_best_weights = True)
vae.compile(optimizer='adam')
# vae.compile(optimizer='adam',metrics = ['accuracy'])
vae.summary()

# train the autoencoder

history=vae.fit(x_train,y_train,
        callbacks=[earlyStop],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test))
# history=vae.fit(my_data1,label_1,
#         validation_split=0.2,
#         callbacks=[earlyStop],
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=(x_test, None))
save_dir='result/2401/3000/34fc_new/2/128_50_1_CMA/1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#loss tra


# print(history.params)
# print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
filename='loss.png'
plt.savefig(save_dir+'/'+filename)
plt.show()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# filename='acc.png'
# plt.savefig(save_dir+'/'+filename)
# plt.show()

vae.save_weights(save_dir+'/vae_ND.h5')

plot_results(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp",save_dir=save_dir)

# decode one point in the latent space
# z_sample = np.array([[-1, 3]])
# x_decoded = decoder.predict(z_sample)
#
# FC = x_decoded[0].reshape(68, 68)
# FC = x_decoded[0].reshape(90, 90)
# plt.figure()
# plt.imshow(FC)


