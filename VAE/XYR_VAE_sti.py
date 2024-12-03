# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 05:05:33 2021

@author: yonis
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from skimage.metrics import structural_similarity as ssim

import pandas as pd
import os
import matplotlib as mpl
import math
from sklearn import preprocessing
import seaborn as sns

from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
#
# x = [1,2,3,4]
# y = [0.848,0.897,0.903,0.900]
# plt.plot(x, y,
#          linewidth=0.5,  
#          linestyle=None,  
#          color='blue',  # 
#          markeredgewidth=0.0,  # 
#          )
# plt.show()


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


def plot_data_in_latent(ax, models,
                        data, color, nombre,
                        batch_size=128):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    # os.makedirs(model_name, exist_ok=True)

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    # plt.figure(figsize=(10, 10))
    #    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    ax.scatter(z_mean[:, 0], z_mean[:, 1], c=color, alpha=0.2, label=nombre)
    # plt.colorbar()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    ax.set_xlabel("Z0", fontsize=15)
    ax.set_ylabel("Z1", fontsize=15)
    # plt.savefig(filename)
    # plt.show()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    # ax.legend()
    return z_mean


# MNIST datase
def plot_pert_in_latent(models,
                        data, color, symbol, plotsi,
                        batch_size=128, 
                        filename='kick'):

    encoder, decoder = models
    x_test, y_test = data
    # os.makedirs(model_name, exist_ok=True)
    # filename = os.path.join(model_name, "vae_mean.png")
    # filename = "2309/0-1_0.02/kick_AD_1.png"
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    # plt.figure(figsize=(12, 10))
    #    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    if plotsi == 1:
        plt.scatter(z_mean[:, 0], z_mean[:, 1], marker=symbol, c=data[1], alpha=0.3, cmap='viridis')
        # plt.colorbar()
        plt.xlabel("Z0", fontsize=15)
        plt.ylabel("Z1", fontsize=15)
        plt.savefig(filename)
        # plt.show()
        # plt.xlim(-4, 4)
        # plt.ylim(-4, 4)
    # print(z_mean)
    return z_mean


def plot_data_kick_in_latent_XYR(models,
                        data1, data2,data3,xc,yc,l,
                        data4, label_kick,distance,  symbol, plotsi,
                        batch_size=128,
                        model_name="vae_mnist", fileName="kick",index=1):

    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    encoder, decoder = models
    x1 = data1
    x2 = data2
    x3 = data3
    #os.makedirs(model_name, exist_ok=True)


    # display a 2D plot of the digit classes in the latent space
    z1_mean, _, _ = encoder.predict(x1, batch_size=batch_size)
    z2_mean, _, _ = encoder.predict(x2, batch_size=batch_size)
    z3_mean, _, _ = encoder.predict(x3, batch_size=batch_size)
    #plt.figure(figsize=(10, 10))
    #    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.scatter(z1_mean[:, 0], z1_mean[:, 1], c='green', alpha=0.5, s=30,linewidths=0)
    plt.scatter(z2_mean[:, 0], z2_mean[:, 1], c='yellow', alpha=0.5,s=30,linewidths=0)
    plt.scatter(z3_mean[:, 0], z3_mean[:, 1], c='red', alpha=0.5,s=30,linewidths=0)
    plt.scatter(xc, yc, c='blue',  marker='D',alpha=1, s=40,linewidths=0)


    confidence = 5.991  # 95%
    make_ellipses(z1_mean, ax, confidence=confidence, color='green', alpha=0.3, eigv=False)
    make_ellipses(z2_mean, ax, confidence=confidence, color='yellow', alpha=0.3, eigv=False)
    make_ellipses(z3_mean, ax, confidence=confidence, color='red', alpha=0.3, eigv=False)
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'm', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen']
    alp = 1
    for sl in range(68):
        kick = np.transpose(data4[:, :, sl])
        kick_data = (kick, label_kick)
        x_test, y_test = kick_data
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test,
                                       batch_size=batch_size)

        z_mean[0, 0] = xc[l]
        z_mean[0, 1] = yc[l]
        # plt.figure(figsize=(12, 10))
        # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)

        if sl >= 34:
            alp = 0.5
            # cl = colors[sl-34]
            cl = 'green'
        else:
            # cl = colors[sl]
            cl = 'blue'
        # if sl in index:
        #     alp = 0.5
        #     cl = colors[sl]
        #
        # else:
        #     # cl = colors[sl]
        #     cl = 'silver'


        x = np.arange(0, z_mean.shape[0], 2)
        if plotsi == 1:
            # plt.scatter(z_mean[0:50:2, 0], z_mean[0:50:2, 1], marker='x', s=20, c=cl, alpha=alp)
            plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', s=20, alpha=1, c=cl)
            plt.plot(z_mean[:, 0], z_mean[:, 1],
                     linewidth=0.3,  # 
                     linestyle=None,  # 
                     color='k',  # 
                     alpha=1-distance[sl]/np.max(distance)+0.001,
                     markeredgewidth=0.0,  #

                     )
    # plt.colorbar()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.xlabel("Z0",fontsize=15)
    plt.ylabel("Z1",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(fileName)
    plt.show()


    return z_mean



def make_ellipses(zz, ax, confidence=5.991, alpha=0.3, color="blue", eigv=False, arrow_color_list=None):

    mean = np.mean(zz, 0)
    cov = np.cov(zz, rowvar=False)
    lambda_, v = np.linalg.eig(cov)  # 
    # print( "lambda: ", lambda_)
    # print("v: ", v)
    # print("v[0, 0]: ", v[0, 0])

    sqrt_lambda = np.sqrt(np.abs(lambda_))  # 

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]  # 
    height = 2 * np.sqrt(s) * sqrt_lambda[1]  # 
     # 
    angle= np.degrees(np.arctan2(*v[:, 0][::-1]))


    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)  # 

    ax.add_artist(ell)
    ell.set_alpha(alpha)
    # 
    if eigv:
        # print "type(v): ", type(v)
        if arrow_color_list is None:
            arrow_color_list = [color for i in range(v.shape[0])]
        for i in range(v.shape[0]):
            v_i = v[:, i]
            scale_variable = np.sqrt(s) * sqrt_lambda[i]
            
            """
            ax.arrow(x, y, dx, dy,    
                     width,    
                     length_includes_head,    
                     head_width,    
                     head_length,   
                     color,   
                     )
            """
            ax.arrow(mean[0], mean[1], scale_variable * v_i[0], scale_variable * v_i[1],
                     width=0.05,
                     length_includes_head=True,
                     head_width=0.2,
                     head_length=0.3,
                     color=arrow_color_list[i])
            # ax.annotate("",
            #             xy=(mean[0] + lambda_[i] * v_i[0], mean[1] + lambda_[i] * v_i[1]),
            #             xytext=(mean[0], mean[1]),
            #             arrowprops=dict(arrowstyle="->", color=arrow_color_list[i]))
    return ell
    # v, w = np.linalg.eigh(cov)
    # print "v: ", v

    # # angle = np.rad2deg(np.arccos(w))
    # u = w[0] / np.linalg.norm(w[0])
    # angle = np.arctan2(u[1], u[0])
    # angle = 180 * angle / np.pi
    # s = 5.991   
    # v = 2.0 * np.sqrt(s) * np.sqrt(v)
    # ell = mpl.patches.Ellipse(xy=mean, width=v[0], height=v[1], angle=180 + angle, color="red")
    # ell.set_clip_box(ax.bbox)
    # ell.set_alpha(0.5)
    # ax.add_artist(ell)


def is_in_ellipse(pt, ell):
   
    x, y = ell.get_center()
    w = ell.get_width()
    h = ell.get_height()
    a = ell.get_angle()

    cos = np.cos(np.radians(180. - a))
    sin = np.sin(np.radians(180. - a))
    xc = pt[0] - x
    yc = pt[1] - y
    xct = xc * cos - yc * sin
    yct = xc * sin + yc * cos
    result = (xct ** 2 / (w / 2.) ** 2) + (yct ** 2 / (h / 2.) ** 2)
    if result <= 1.0:
        return True
    else:
        return False


def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    # return normData, ranges, minVals, maxVals
    return normData


def plot_SSIM_for_each_stipara(ssim_index, sim_index, amplitude):
    n = math.ceil(math.sqrt(len(sim_index)))
    fig, ax = plt.subplots(n, n, figsize=(6, 6))
    k = 0
    for i in range(n):
        if k >= len(sim_index):
            break
        for j in range(n):
            if k >= len(sim_index):
                break
            y = ssim_index[sim_index[k]]
            x = amplitude
            t = sim_index[k] + 1
            ax[i][j].plot(x, y)
            ax[i][j].set_title(t, fontsize=5)
            ax[i][j].set_xlabel('amplitude', fontsize=5)
            ax[i][j].set_ylabel('SSIM', fontsize=5)
            ax[i][j].tick_params(labelsize=5)
            k = k + 1


def plot_MateSSIM_for_each_stipara(mate, ssim_index, sim_index, amplitude):
    # mate= noramlization(mate)

    ssim_index_n = noramlization(ssim_index)

    n = math.ceil(math.sqrt(len(ssim_index)))
    fig, ax = plt.subplots(n, n, figsize=(6, 6))
    k = 0
    for i in range(n):
        if k >= len(sim_index):
            break
        for j in range(n):
            if k >= len(sim_index):
                break
            min_max_scaler = preprocessing.MinMaxScaler()
            y1 = ssim_index_n[k]
            y2 = mate[sim_index[k]]

            x = amplitude
            t = sim_index[k] + 1
            ax[i][j].plot(x, y1, color='blue')
            ax[i][j].plot(x, y2, color='red')
            ax[i][j].set_title(t, fontsize=5)
            ax[i][j].set_xlabel('amplitude', fontsize=5)
            ax[i][j].set_ylabel('SSIM', fontsize=5)
            ax[i][j].tick_params(labelsize=5)
            k = k + 1
def plot_part_in_latent(models,
                    decode_fc_cn,decode_fc_mci,decode_fc_ad):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)

    # z = []
    part=np.zeros((20,20))
    for i in range(len(x)):
        for j in range(len(y)):
            # z.append([x[i], y[j]])
            out = decoder.predict([[x[i], y[j]]])
    # for i in range(len(out)):
    #     outi=np.array(out[i])
            sim1 = ssim(out.reshape(68, 68), decode_fc_cn.reshape(68, 68))
            sim2 = ssim(out.reshape(68, 68), decode_fc_mci.reshape(68, 68))
            sim3 = ssim(out.reshape(68, 68), decode_fc_ad.reshape(68, 68))
            if sim1 > sim2 and sim1 > sim3:
                # part.append(1)
                part[19-j][i] = 1
            elif sim1 < sim2 and sim2 > sim3:
                # part.append(2)
                part[19-j][i] = 2
            else:
                # part.append(3)
                part[19-j][i] = 3
    ax = plt.subplots(figsize=(6,6))
    ax= sns.heatmap(part,cmap='RdYlGn_r')
        # plt.fill_between(x, y, where=sim1 > sim2 and sim1 > sim3, color='green')
        # plt.fill_between(x, y, where=sim1 < sim2 and sim2 > sim3, color='yellow')
        # plt.fill_between(x, y, where=sim1 < sim3 and sim2 < sim3, color='red')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    return None

def plot_results(models,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models

    """

    encoder, decoder = models
    # x_test, y_test = data
    # os.makedirs(model_name, exist_ok=True)
    #
    # filename = os.path.join(model_name, "vae_mean.png")
    # # display a 2D plot of the FC classes in the latent space
    # z_mean, _, _ = encoder.predict(x_test,
    #                                batch_size=batch_size)
    # plt.figure(figsize=(12, 10))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.xlim(-4,4)
    # plt.ylim(-4,4)
    # plt.savefig(filename)
    # plt.show()

    filename = 'result/FCs_over_latent5.png'
    # display a 30x30 2D manifold of FCs
    n = 24
    digit_size = 68
    figure = np.zeros((digit_size * n, digit_size * n))
    left_hemisphere_mean = np.zeros((n, n))
    right_hemisphere_mean = np.zeros((n, n))
    interhemispheric_mean = np.zeros((n, n))
    fc_mean = np.zeros((n, n))
    fc_LI = np.zeros((n, n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of FC classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
   
    half_nodes = digit_size // 2

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

            left_hemisphere = digit[:half_nodes, :half_nodes]
            right_hemisphere = digit[half_nodes:, half_nodes:]
            #
           
            left_hemisphere_mean[i,j] = np.mean(left_hemisphere)
            right_hemisphere_mean[i,j] = np.mean(right_hemisphere)
            #
            
            interhemispheric_mean[i,j] = np.mean(digit[:half_nodes, half_nodes:])
            
            fc_mean[i,j] = np.mean(digit)
           
            FCT = np.mean(digit, axis=1)
            fc_LI[i,j]=(np.sum(FCT[1:34])-np.sum(FCT[34:68]))/(np.sum(FCT[1:34])+np.sum(FCT[34:68]));

    start_range = 0
    end_range = n * 1
    pixel_range = np.arange(start_range, end_range, 1)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    # plt.figure(figsize=(20, 20))
    # filename = 'result/left_hemisphere_over_latent1.png'
    # plt.imshow(left_hemisphere_mean)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    #
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.savefig(filename)
    # plt.show()
    #
    # plt.figure(figsize=(20, 20))
    # filename = 'result/right_hemisphere_over_latent1.png'
    # plt.imshow(right_hemisphere_mean)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.savefig(filename)
    # plt.show()
    #
    # plt.figure(figsize=(20, 20))
    # filename = 'result/interhemispheric_mean_over_latent1.png'
    # plt.imshow(interhemispheric_mean)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.savefig(filename)
    # plt.show()

    # plt.figure(figsize=(20, 20))
    # filename = 'result/fc_mean_over_latent1.png'
    # plt.imshow(fc_mean)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.xticks(sample_range_x)
    # plt.yticks(sample_range_y)
    # plt.savefig(filename)
    # plt.show()

    # plt.figure(figsize=(20, 20))
    filename = 'result/fc_LI_over_latent3.png'
    plt.imshow(fc_LI,cmap='coolwarm')
    plt.xlabel("Z0", fontsize=15)
    plt.ylabel("Z1", fontsize=15)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.savefig(filename)
    plt.show()

    # pd.DataFrame(left_hemisphere_mean).to_csv('result/2309/0-1_2/left_meanfc1.csv')
    # pd.DataFrame(right_hemisphere_mean).to_csv('result/2309/0-1_2/right_meanfcright_meanfc1.csv')
    # pd.DataFrame(interhemispheric_mean).to_csv('result/2309/0-1_2/inter_meanfc1.csv')
    # pd.DataFrame(fc_mean).to_csv('result/2309/0-1_2/all_meanfc1.csv')
    # pd.DataFrame(fc_LI).to_csv('result/2309/0-1_2/all_LIfc1.csv')
    # plt.figure(figsize=(8, 8))
    # start_range = digit_size // 2
    # end_range = (n - 1) * digit_size + start_range + 1
    # start_range = 0
    # end_range = n * digit_size
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.xlabel("Z0",fontsize=15)
    # plt.ylabel("Z1",fontsize=15)
    # plt.imshow(figure) #cmap='Greys_r')
    # plt.colorbar()
    # plt.grid(color='w', linewidth=2)
    # plt.savefig(filename)
    # plt.show()

def plot_data_kick_in_latent_distinct_XYR(models,
                        data1, data2,data3,xc,yc,l,
                        data4, label_kick,distance,
                        batch_size=128,save_dir='result'
                        ):

    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x1 = data1
    x2 = data2
    x3 = data3
    #os.makedirs(model_name, exist_ok=True)


    # display a 2D plot of the digit classes in the latent space
    z1_mean, _, _ = encoder.predict(x1, batch_size=batch_size)
    z2_mean, _, _ = encoder.predict(x2, batch_size=batch_size)
    z3_mean, _, _ = encoder.predict(x3, batch_size=batch_size)

    for sl in range(34):
        plt.figure(figsize=(6, 6))
        fig, ax = plt.subplots()
        plt.scatter(z1_mean[:, 0], z1_mean[:, 1], c='green', alpha=0.5, s=30, linewidths=0)
        plt.scatter(z2_mean[:, 0], z2_mean[:, 1], c='yellow', alpha=0.5, s=30, linewidths=0)
        plt.scatter(z3_mean[:, 0], z3_mean[:, 1], c='red', alpha=0.5, s=30, linewidths=0)
        plt.scatter(xc, yc, c='blue', marker='D', alpha=1, s=40, linewidths=0)

        confidence = 5.991  # 95%
        make_ellipses(z1_mean, ax, confidence=confidence, color='green', alpha=0.3, eigv=False)
        make_ellipses(z2_mean, ax, confidence=confidence, color='yellow', alpha=0.3, eigv=False)
        make_ellipses(z3_mean, ax, confidence=confidence, color='red', alpha=0.3, eigv=False)
        colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
              'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
              'cornflowerblue', 'cornsilk', 'crimson', 'm', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
              'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
              'darksalmon', 'darkseagreen']

        kick = np.transpose(data4[:, :, sl])
        kick_data = (kick, label_kick)
        x_test, y_test = kick_data
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
        z_mean[0, 0] = xc[l]
        z_mean[0, 1] = yc[l]


        kick2 = np.transpose(data4[:, :, sl + 34])
        kick_data2 = (kick2, label_kick)
        x_test2, y_test2 = kick_data2
        z_mean2, _, _ = encoder.predict(x_test2, batch_size=batch_size)
        z_mean2[0, 0] = xc[l]
        z_mean2[0, 1] = yc[l]

        x = np.arange(0, z_mean.shape[0], 2)

        plt.scatter(z_mean[0:50:2, 0], z_mean[0:50:2, 1], marker='x', s=20, c=colors[sl], alpha=1)

        plt.plot(z_mean[0:50:2, 0], z_mean[0:50:2, 1],
            linewidth=0.3,  
            linestyle=None,  
            color='k',  
            alpha=1 - distance[sl] / np.max(distance) + 0.001,
            markeredgewidth=0.0, 
         )
        plt.scatter(z_mean2[0:50:2, 0], z_mean2[0:50:2, 1], marker='x', s=20, c=colors[sl], alpha=0.5)

        plt.plot(z_mean2[0:50:2, 0], z_mean2[0:50:2, 1],
            linewidth=0.3, 
            linestyle=None,  
            color='k',  
            alpha=1 - distance[sl + 34] / np.max(distance) + 0.001,
            markeredgewidth=0.0,  
         )
        plt.xlabel("Z0", fontsize=15)
        plt.ylabel("Z1", fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

        plt.savefig(save_dir + '/2sti_dist{}.png'.format(sl))
        plt.clf()  




# ----Encodear las matrices modeladas y ver qué salen---#
# define el modelo

original_dim = 4624
# network parameters
input_shape = (original_dim,)
intermediate_dim = 1028
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
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
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
# carga los pesos ya entrenados
vae.load_weights('D:/yan/code/VAEcode2/step1_train/result/2401/3000/34fc_new/128_50_1_CMA/2/vae_ND.h5', by_name=True)

# # plot fc cover latent space without state circle
# plot_results(models,
#             batch_size=batch_size,
#             model_name="vae_mlp")

# EMP
# fc = scipy.io.loadmat('data_emp/FC_avg.mat')
# FC = np.array(fc['FC_avg'])
# plt.figure(figsize=(6, 6))
# #label_CN = np.ones(100) * 1
# label_CN = np.array([1,2,3])
# CN_data = (FC, label_CN)
# zz = plot_data_in_latent(models,
#                          CN_data, ['green','yellow','red'], ['CN','MCI','AD'],
#                          batch_size=batch_size,
#                          model_name="vae_mlp")
# plt.savefig('result_emp/VAE_emp_avg.png')
# plt.show()
#

# sim
# load no sti data
cn = scipy.io.loadmat('data/2401/34fc_new/CN_100.mat')
CN = np.array(cn['FC_CN'])
ad = scipy.io.loadmat('data/2401/34fc_new/AD_100.mat')
AD = np.array(ad['FC_AD'])
mci = scipy.io.loadmat('data/2401/34fc_new/MCI_100.mat')
MCI = np.array(mci['FC_MCI'])

# # get FC matriz after decoder
# z_sample = np.array([[1, 1]])
# x_decoded = decoder.predict(z_sample)
# FC = x_decoded[0].reshape(68, 68)
# plt.figure()
# plt.imshow(FC)
# plt.colorbar()
# plt.savefig('result/decoder_fc.png')
#
# ##emp and decoder
# z_sample, _, _ = encoder.predict(CN[0:100, :], batch_size=batch_size)
# x_decoded = decoder.predict(z_sample)
# FC = CN[1, :].reshape(68, 68)
# FC_de = x_decoded[1].reshape(68, 68)
#
# upper_indices = np.triu_indices(68, k = 1)
# Arr = FC[upper_indices]
# # Do stuff with Arr
# FC_de[upper_indices] =Arr
# plt.figure()
# plt.imshow(FC_de)
# plt.colorbar()
# plt.savefig('result/cn_fc.png')
# plt.show()
#
# z_sample, _, _ = encoder.predict(MCI[0:100, :], batch_size=batch_size)
# x_decoded = decoder.predict(z_sample)
# FC = MCI[1, :].reshape(68, 68)
# FC_de = x_decoded[1].reshape(68, 68)
#
# upper_indices = np.triu_indices(68, k = 1)
# Arr = FC[upper_indices]
# # Do stuff with Arr
# FC_de[upper_indices] =Arr
# plt.figure()
# plt.imshow(FC_de)
# plt.colorbar()
# plt.savefig('result/mci_fc.png')
# plt.show()
# z_sample, _, _ = encoder.predict(AD[0:100, :], batch_size=batch_size)
# x_decoded = decoder.predict(z_sample)
# FC = AD[1, :].reshape(68, 68)
# FC_de = x_decoded[1].reshape(68, 68)
#
# upper_indices = np.triu_indices(68, k = 1)
# Arr = FC[upper_indices]
# # Do stuff with Arr
# FC_de[upper_indices] =Arr
# plt.figure()
# plt.imshow(FC_de)
# plt.colorbar()
# plt.savefig('result/ad_fc.png')
# plt.show()

# hace la figura con todos los puntos encodeados


CN_short = CN[0:100, :]
AD_short = AD[0:100, :]
MCI_short = MCI[0:100, :]

# plt.figure(figsize=(6, 6))
fig, ax = plt.subplots(figsize=(6, 6))
label_CN = np.ones(100) * 0
CN_data = (CN[0:100, :], label_CN)
zz = plot_data_in_latent(ax, models, CN_data, 'green', 'CN', batch_size=batch_size)
CN_centroid = np.mean(zz, 0)
# tandard DeviationalEllipse
confidence = 5.991  # 95%
ell_cn = make_ellipses(zz, ax, confidence=confidence, color='green', alpha=0.1, eigv=False)
de_CN = decoder.predict(CN_centroid.reshape(1, 2))
distanceCN2CN = np.sqrt((zz[:,0] - CN_centroid[0]) ** 2 + (zz[:,1] - CN_centroid[1]) ** 2)

label_AD = np.ones(100) * 2
AD_data = (AD[0:100, :], label_AD)
zz = plot_data_in_latent(ax, models, AD_data, 'red', 'AD', batch_size=batch_size)
AD_centroid = np.mean(zz, 0)
make_ellipses(zz, ax, confidence=confidence, color='red', alpha=0.2, eigv=False)
distanceAD2CN = np.sqrt((zz[:,0] - CN_centroid[0]) ** 2 + (zz[:,1] - CN_centroid[1]) ** 2)
# pd.DataFrame(distanceCN2CN).to_csv('F:/XYR/Jean_420/step3_analysis/mulproperty/distanceCN2CN.csv')
# pd.DataFrame(distanceAD2CN).to_csv('F:/XYR/Jean_420/step3_analysis/mulproperty/distanceAD2CN.csv')


label_MCI = np.ones(100) * 1
MCI_data = (MCI[0:100, :], label_MCI)
zz = plot_data_in_latent(ax, models, MCI_data, 'yellow', 'MCI', batch_size=batch_size)
MCI_centroid = np.mean(zz, 0)
make_ellipses(zz, ax, confidence=confidence, color='yellow', alpha=0.2, eigv=False)
x_cent = [CN_centroid[0], MCI_centroid[0], AD_centroid[0]]
y_cent = [CN_centroid[1], MCI_centroid[1], AD_centroid[1]]
# distanceAD2CN = np.sqrt((AD_centroid[0] - CN_centroid[0]) ** 2 + (AD_centroid[1] - CN_centroid[1]) ** 2)


# plt.scatter(CN_centroid[0], CN_centroid[1], c='darkgreen', marker='D', alpha=1, s=100, linewidths=0)
# plt.scatter(MCI_centroid[0], MCI_centroid[1], c='darkorange', marker='D', alpha=1, s=100, linewidths=0)
# plt.scatter(AD_centroid[0], AD_centroid[1], c='darkred', marker='D', alpha=1, s=100, linewidths=0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# filename = 'result/2401/3000/34fc_final/128_50_2/weight2/0-1_0.02/dist'
# if not os.path.exists(filename):
#     os.makedirs(filename)
# plt.savefig(filename+'dis_state.png')
# plt.show()


# #plot fc over latent space with state circle
n =24
digit_size = 68
figure = np.zeros((digit_size * n, digit_size * n))
left_hemisphere_mean = np.zeros((n, n))
right_hemisphere_mean = np.zeros((n, n))
interhemispheric_mean = np.zeros((n, n))
fc_mean = np.zeros((n, n))
fc_LI = np.zeros((n, n))
# linearly spaced coordinates corresponding to the 2D plot
# of FC classes in the latent space
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]
# 
half_nodes = digit_size // 2

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit
        FCT = np.mean(digit, axis=1)
        fc_LI[i, j] = (np.sum(FCT[1:34]) - np.sum(FCT[34:68])) / (np.sum(FCT[1:34]) + np.sum(FCT[34:68]));

start_range = 0
end_range = n * 1
pixel_range = np.arange(start_range, end_range, 1)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)

clist = ['black', 'dodgerblue']
newcmp = LinearSegmentedColormap.from_list('chaos', clist)

# plt.imshow(figure, extent=[-3, 3, -3, 3], cmap='Blues', alpha=1)
plt.imshow(fc_LI, extent=[-3, 3, -3, 3], cmap='coolwarm', alpha=1)
plt.xlabel("Z0", fontsize=15)
plt.ylabel("Z1", fontsize=15)
plt.colorbar()
# # filename = 'result/2401/3000/34fc_new/128_50_2/weight2/0-1_0.02/fc_LI_latent2.png'
# # plt.savefig(filename)
# # plt.show()
#
#
# # state change start
# plt.xlim(-3,3)
# plt.ylim(-3,3)
#
xvals_sub = np.linspace(AD_centroid[0], MCI_centroid[0], 4)
yinterp_sub = np.linspace(AD_centroid[1], MCI_centroid[1], 4)
# yinterp_sub = np.interp(xvals_sub, x_cent, y_cent)

xvals_sub2 = np.linspace(MCI_centroid[0], CN_centroid[0], 5)
yinterp_sub2 = np.linspace(MCI_centroid[1], CN_centroid[1], 5)
# yinterp_sub2 = np.interp(xvals_sub2, x_cent, y_cent)

xvals = np.linspace(AD_centroid[0], MCI_centroid[0], 100)
yinterp = np.linspace(AD_centroid[1], MCI_centroid[1], 100)
# yinterp = np.interp(xvals, x_cent, y_cent)
xvals2 = np.linspace(MCI_centroid[0], CN_centroid[0], 100)
yinterp2 = np.linspace(MCI_centroid[1], CN_centroid[1], 100)
# yinterp2 = np.interp(xvals2, x_cent, y_cent)
#
from matplotlib.collections import LineCollection


def color_map(data, cmap):
    

    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256 / cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255 * (data - dmin) / (dmax - dmin))

    return cs[data]


ps = np.stack((xvals, yinterp), axis=1)
segments = np.stack((ps[:-1], ps[1:]), axis=1)
clist = ['orange', 'darkred']
newcmp = LinearSegmentedColormap.from_list('chaos', clist)
colors = color_map(yinterp[:-1], newcmp)
line_segments = LineCollection(segments, colors=colors, linewidths=3, linestyles='solid', cmap=newcmp,alpha=1)
line_segments.set_linewidth(8)
line_segments.set_zorder(0)
# fig, ax = plt.subplots()
ax.add_collection(line_segments)

ps = np.stack((xvals2, yinterp2), axis=1)
segments = np.stack((ps[:-1], ps[1:]), axis=1)
clist = ['green', 'orange']
newcmp = LinearSegmentedColormap.from_list('chaos', clist)
colors = color_map(yinterp2[:-1], newcmp)
line_segments = LineCollection(segments, colors=colors, linewidths=3, linestyles='solid', cmap=newcmp,alpha=1)
line_segments.set_linewidth(8)
line_segments.set_zorder(0)
# fig, ax = plt.subplots()
ax.add_collection(line_segments)
# plt.show()
#arrow

arrowprops = {
    'arrowstyle': '->',
    'linewidth': 5,
    'color': 'red'

}

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
x_tail=xvals[8]
y_tail =yinterp[8]
x_head=xvals[0]
y_head =yinterp[0]
dx= x_head-x_tail
dy = y_head-y_tail

arrow =  mpatches.FancyArrow(x_tail, y_tail, dx, dy,
                            width=0, head_width=0.5,length_includes_head=True,facecolor='darkred', edgecolor='darkred', alpha=0.8)
# arrow = mpatches.FancyArrowPatch((xvals[10],yinterp[10]), (xvals[0],yinterp[0]),
#                                  mutation_scale=10,
#                                  transform=ax.transAxes)
ax.add_patch(arrow)

# plt.annotate("", xy=[xvals[0],yinterp[0]], xytext=[xvals[10],yinterp[10]], arrowprops=arrowprops)

plt.scatter(xvals_sub[1:3],yinterp_sub[1:3],s=100,marker='o',color='black')
plt.scatter(xvals_sub2[1:4],yinterp_sub2[1:4],s=100,marker='o',color='black')
# plt.scatter(xvals_sub,yinterp_sub,s=100,marker='D',color=['darkred','red','hotpink','saddlebrown'])
# plt.scatter(xvals_sub2,yinterp_sub2,s=100,marker='D',color=['saddlebrown','darkgoldenrod','dodgerblue','lime','green'])

plt.scatter(xvals_sub2[4],yinterp_sub2[4],s=100,marker='o',color='green')
plt.scatter(xvals_sub2[0],yinterp_sub2[0],s=100,marker='o',color='orange')
plt.scatter(xvals_sub[0],yinterp_sub[0],s=100,marker='o',color='darkred')

bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Z1', fontsize=15)
plt.xlabel('Z0', fontsize=15)

# plt.legend(prop={'size' : 15})
plt.savefig('result/2401/3000/34fc_new/128_50_2/weight2/0-1_0.02/statechange4.png',dpi=100)
plt.show()
# # plt.show()
# #
# #
# #
# matriz
for i in range(len(xvals_sub2)):
    z_sample = np.array([[xvals_sub2[i], yinterp_sub2[i]]])
    x_decoded = decoder.predict(z_sample)
    FC = x_decoded[0].reshape(68, 68)
    plt.figure()
    bwith= 5  
    # if i==0:
    #     bc='saddlebrown'
    # elif i==1:
    #     bc = 'darkgoldenrod'
    # elif i==2:
    #     bc = 'dodgerblue'
    # elif i==3:
    #
    #     bc = 'lime'
    #
    # else:
    #     bc = 'darkgreen'
    bc = 'yellow'
   # clist = ['black', bc]

    # clist = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#1B9E8A', '#32B67B', '#6CCC5F', '#B4DD3D', '#FDE73A']
    clist = ['#000000', '#800000','#FF0000','#1B9E8A', '#6CCC5F', '#B4DD3D', '#FDE73A','#FFFF00']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist)
    plt.imshow(FC,cmap=newcmp,alpha=1)
    plt.colorbar()
    a = 'decoder_15_fc_MC_%f_2.png' % i
    plt.xticks([])
    plt.yticks([])
    # ax = plt.gca()  
    # ax.spines['top'].set_color(bc)  
    # ax.spines['bottom'].set_color(bc)
    # ax.spines['right'].set_color(bc)
    # ax.spines['left'].set_color(bc)
    #
    # ax.spines['bottom'].set_linewidth(bwith)
    # ax.spines['left'].set_linewidth(bwith)
    # ax.spines['top'].set_linewidth(bwith)
    # ax.spines['right'].set_linewidth(bwith)

    plt.savefig('result/2401/3000/34fc_new/128_50_2/weight2/'+a)
    plt.show()
for i in range(len(xvals_sub)):
    z_sample = np.array([[xvals_sub[i], yinterp_sub[i]]])
    x_decoded = decoder.predict(z_sample)
    FC = x_decoded[0].reshape(68, 68)
    plt.figure()
    bwith= 5 
    # if i==0:
    #
    #     bc='darkred'
    # elif i==1:
    #
    #     bc = 'red'
    # elif i==2:
    #
    #     bc = 'hotpink'
    # else:
    #
    #     bc = 'orange'
    bc = 'yellow'
    # clist = ['black', bc]
    clist = ['#000000', '#800000','#FF0000' ,'#1B9E8A', '#6CCC5F', '#B4DD3D', '#FDE73A',  '#FFFF00']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist)
    plt.imshow(FC, cmap=newcmp,alpha=1)
    plt.colorbar()
    a = 'decoder_15_fc_AM_%f.png' % i
    plt.xticks([])
    plt.yticks([])

    # ax = plt.gca() 
    # ax.spines['top'].set_color(bc)  
    # ax.spines['bottom'].set_color(bc)
    # ax.spines['right'].set_color(bc)
    # ax.spines['left'].set_color(bc)
    #
    # ax.spines['bottom'].set_linewidth(bwith)
    # ax.spines['left'].set_linewidth(bwith)
    # ax.spines['top'].set_linewidth(bwith)
    # ax.spines['right'].set_linewidth(bwith)

    plt.savefig('result/2401/3000/34fc_new/128_50_2/weight2/'+a)
    plt.show()
# #
# # # #decode cent
# # # decode_fc_cn = decoder.predict(CN_centroid.reshape((1, 2)))
# # # decode_fc_mci = decoder.predict(MCI_centroid.reshape((1, 2)))
# # # decode_fc_ad = decoder.predict(AD_centroid.reshape((1, 2)))
# # # plt.figure(figsize=(6, 6))
# # # plot_part_in_latent(models, decode_fc_cn, decode_fc_mci, decode_fc_ad)
# # # plt.savefig('result/space_partion.png')
# # # plt.show()
# #
# # # importa todas las perturbaciones

# # # mat = scipy.io.loadmat('data3/kick_CN_full_10hz.mat')
# # # kick_full_CN = mat['FC_sim_kick_CN']
# #
# # # data_dir = 'data/0.2_130_0.0002_480'
# # # save_dir = 'result/0.2_130_0.0002_480'
# # # if not os.path.exists(save_dir):
# # #     os.makedirs(save_dir)
# # #
# # # mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim.mat')
# # # kick_full_AD = mat['FC_sim_kick_AD']
# # # # mat = scipy.io.loadmat(data_dir+'/kick_MCI_FC_sim.mat')
# # # # kick_full_MCI = mat['FC_sim_kick_MCI']
# # #
# # # numofP = kick_full_AD.shape[1]
# # # label_kick = 1
# # #
# # # kick_AD = kick_full_AD[:, 0:numofP, :]
# # # # kick_AD = noramlization(kick_AD)
# # # ssim_index = []
# # # for kk in range(68):
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick, batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', c='b', alpha=0.8)
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # # plt.savefig(save_dir + '/only_sti_AD.png')
# # # plt.show()
# #
# # # #try
# # # data_dir='data/120-0.02'
# # # save_dir='result/120-0.02'
# # # if not os.path.exists(save_dir):
# # #     os.makedirs(save_dir)
# # #
# # # mat = scipy.io.loadmat(data_dir+'/i_d/kick_AD_FC_sim_to1.mat')
# # # kick_full_AD = mat['FC_AD']
# # # mat = scipy.io.loadmat(data_dir+'/i_d/kick_MCI_FC_sim_to1.mat')
# # # kick_full_MCI = mat['FC_MCI']
# # #
# # # numofP=kick_full_AD.shape[1]
# # # label_kick = 1
# # #
# # # kick_AD = kick_full_AD[:,0:numofP,:]
# # # # kick_AD = noramlization(kick_AD)
# # # ssim_index=[]
# # # for kk in range(68):
# # #
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick,batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x',  c='b', alpha=0.8)
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # # mat = scipy.io.loadmat(data_dir+'/i_p/kick_AD_FC_sim_to1.mat')
# # # kick_full_AD = mat['FC_AD']
# # # mat = scipy.io.loadmat(data_dir+'/i_p/kick_MCI_FC_sim_to1.mat')
# # # kick_full_MCI = mat['FC_MCI']
# # #
# # # numofP=kick_full_AD.shape[1]
# # # label_kick = 2
# # #
# # # kick_AD = kick_full_AD[:,0:numofP,:]
# # #
# # # for kk in range(68):
# # #
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick,batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', c='c', alpha=0.8 )
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # # mat = scipy.io.loadmat(data_dir+'/s_d/kick_AD_FC_sim_to1.mat')
# # # kick_full_AD = mat['FC_AD']
# # # mat = scipy.io.loadmat(data_dir+'/s_d/kick_MCI_FC_sim_to1.mat')
# # # kick_full_MCI = mat['FC_MCI']
# # #
# # # numofP=kick_full_AD.shape[1]
# # # label_kick = 3
# # #
# # # kick_AD = kick_full_AD[:,0:numofP,:]
# # #
# # # for kk in range(68):
# # #
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick,batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', c='r', alpha=0.8)
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # # mat = scipy.io.loadmat(data_dir+'/s_p/kick_AD_FC_sim_to1.mat')
# # # kick_full_AD = mat['FC_AD']
# # # mat = scipy.io.loadmat(data_dir+'/s_p/kick_MCI_FC_sim_to1.mat')
# # # kick_full_MCI = mat['FC_MCI']
# # #
# # # numofP=kick_full_AD.shape[1]
# # # label_kick = 1
# # #
# # # kick_AD = kick_full_AD[:,0:numofP,:]
# # #
# # # for kk in range(68):
# # #
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick,batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x',  c='m', alpha=0.8)
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # #
# # # mat = scipy.io.loadmat(data_dir+'/24-2/kick_AD_FC_sim_to1.mat')
# # # kick_full_AD = mat['FC_AD']
# # #
# # # numofP=kick_full_AD.shape[1]
# # # label_kick = 1
# # #
# # # kick_AD = kick_full_AD[:,0:numofP,:]
# # #
# # # for kk in range(68):
# # #
# # #     kick = np.transpose(kick_AD[:, :, kk])
# # #     kick_data = (kick, label_kick)
# # #     encoder, decoder = models
# # #     z_mean, _, _ = encoder.predict(kick,batch_size=batch_size)
# # #     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x',  c='g', alpha=0.8)
# # #     tm = np.array(z_mean)
# # #     zzd = decoder.predict(tm.reshape(1, 2))
# # #     sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
# # #     ssim_index.append(sim1)
# # #
# # # ssim_index = np.array(ssim_index).reshape(-1 , 5)
# # #
# # # plt.savefig(save_dir+'/sti_AD.png')
# # # plt.show()
# # #
# # # amplitude = np.arange(1, 68.1, 1)
# # # plt.xlabel('脑区')  
# # # plt.ylabel('SSIM')  
# # #
# # # plt.scatter(amplitude, ssim_index[:, 0], marker='x',  c='b') 
# # # plt.scatter(amplitude, ssim_index[:, 1], marker='x',  c='c')
# # # plt.scatter(amplitude, ssim_index[:, 2], marker='x',  c='r')
# # # plt.scatter(amplitude, ssim_index[:, 3], marker='x',  c='m')
# # # plt.scatter(amplitude, ssim_index[:, 4], marker='x',  c='g')
# # # plt.legend(['i_d', 'i_p', 's_d', 's_p', 'i_d:24-2'])
# # # plt.savefig(save_dir+'/SSIM.png')
# # # plt.show()
# #
# # # kick

data_dir = 'data/2401/34fc_new/result_group_sti_0-1_0.02_f10_90_120_34fc_6p2_--'
save_dir ='result/2401/3000/34fc_final/128_50_2/weight2/result_6p2_--'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#
for kii in range(1):
    ki = kii+1
    mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_%d'%(ki)+'.mat')
    kick_full_AD = mat['FC_sim_kick_AD']
    # mat = scipy.io.loadmat(data_dir + '/kick_MCI_FC_sim_avg.mat')
    # kick_full_MCI = mat['FC_sim_kick_MCI']

   
    numofP = kick_full_AD.shape[1]
    numofP2 = numofP
    label_kick = np.arange(0, numofP2)
    numStil=68

    #  save ssim
    kick_AD = kick_full_AD[:, 0:numofP, :]
    zz = []
    Z = np.zeros((numofP2,latent_dim,numStil))
    dist_full_nodesW = np.zeros((2, numStil))
    dist_kick_node2C_AD = np.zeros((numofP2, numStil))
    dist_last_node2C = []
    # plt.figure(figsize=(6, 6))
    sim_index = []
    ssim_index = np.zeros((numStil,numofP2))
    dist = np.zeros((numStil,numofP2))
    for kk in range(numStil):

        kick = np.transpose(kick_AD[:, :, kk])
        kick_data = (kick, label_kick)
        zz1 = plot_pert_in_latent(models,
                                  kick_data, 'm', 'x', 1,
                                  batch_size=batch_size,
                                  filename=save_dir + '/only_sti_AD_%d'%(ki)+'.png')
        Z[:, :, kk] = zz1
        # print(len(zz1))
        flag = True
        for i in range(len(zz1)):
            tm = np.array(zz1[i, :])
            zzd = decoder.predict(tm.reshape(1, 2))
            sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
            ssim_index[kk, i] = sim1
            dis1 = np.sqrt((tm[0] - CN_centroid[0]) ** 2 + (tm[1] - CN_centroid[1]) ** 2)
            dist[kk, i] = dis1
            if is_in_ellipse(tm, ell_cn) and flag:
                sim_index.append(kk)
                flag = False

        enco_pert = zz1[numofP2 - 1, :]
        zz.append(enco_pert)

        distance = np.sqrt((enco_pert[0] - CN_centroid[0]) ** 2 + (enco_pert[1] - CN_centroid[1]) ** 2)
        dist_last_node2C.append(distance)
        for i in range(numofP2):
            dist_kick_node2C_AD[i, kk] = np.sqrt((zz1[i, 0] - CN_centroid[0]) ** 2 + (zz1[i, 1] - CN_centroid[1]) ** 2)
    # plt.savefig(save_dir + '/only_sti_AD.png')
    # plt.show()
    np.savetxt(save_dir + '/sim_index_AD_%d'%(ki)+'.txt', sim_index)
    pd.DataFrame(ssim_index).to_csv(save_dir + '/SSIM_AD_%d'%(ki)+'.csv')
    pd.DataFrame(dist).to_csv(save_dir + '/dis_AD_%d'%(ki)+'.csv')


#
# amplitude = np.arange(0.02, 2.01, 0.02)

# me = scipy.io.loadmat(data_dir + '/kick_AD_metastable_sim_avg_bold90.mat')
# metastable = np.squeeze(np.array(me['metastable_sim_kick_AD_bold90']).T)[:, 0:numofP]

# plot_MateSSIM_for_each_stipara(metastable, ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM-meta_for_amplitude_AD.png')
# plt.show()

# plot_SSIM_for_each_stipara(ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM_for_amplitude_AD.png')
# plt.show()

pert_AD = np.asarray(zz)
pert_AD_centroid = np.mean(pert_AD, 0)
dist_last_node2C = np.array(dist_last_node2C)
dist_full_nodesW[0, :] = dist_last_node2C
index = []  
index = np.argsort(dist_full_nodesW[0])
print(dist_full_nodesW[0, index])
print(index)
np.savetxt(save_dir + '/dist_kick_index_AD_1.txt', index)

if not os.path.exists(save_dir):
     os.makedirs(save_dir)
# plot_data_kick_in_latent_distinct_XYR(models,
#                                       CN[0:100, :], MCI[0:100, :], AD[0:100, :], x_cent, y_cent, 2,
#                                       kick_AD, label_kick, dist_full_nodesW[0, :],
#                                       batch_size=batch_size,save_dir = save_dir+"/stidis")


# plt.figure(figsize=(6, 6))
XY2 = plot_data_kick_in_latent_XYR(models,
                              CN[0:100, :],MCI[0:100, :],AD[0:100, :],x_cent,y_cent,2,
                              kick_AD,label_kick, dist_full_nodesW[0, :],
                               'x', 1,
                              batch_size=batch_size,
                              model_name="vae_mlp",fileName=save_dir + "/kick_AD_10hz_lr1.png",index=sim_index)

# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_2.mat')
# kick_full_AD = mat['FC_sim_kick_AD']
# # mat = scipy.io.loadmat(data_dir + '/kick_MCI_FC_sim.mat')
# # kick_full_MCI = mat['FC_sim_kick_MCI']
#
# #  save ssim
# kick_AD = kick_full_AD[:, 0:numofP, :]
# zz = []
#
# dist_full_nodesW = np.zeros((2, numStil))
# dist_kick_node2C_AD = np.zeros((numofP, numStil))
# dist_last_node2C = []
# # plt.figure(figsize=(6, 6))
# sim_index = []
# ssim_index = np.zeros((numStil,numofP))
# dist = np.zeros((numStil,numofP))
# for kk in range(numStil):
#
#     kick = np.transpose(kick_AD[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz1 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size,
#                               filename=save_dir + '/only_sti_AD_2.png')
#     Z[:, :, kk] = (Z[:, :, kk]+zz1)/2
#     # print(len(zz1))
#     flag = True
#     for i in range(len(zz1)):
#         tm = np.array(zz1[i, :])
#         zzd = decoder.predict(tm.reshape(1, 2))
#         sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
#         ssim_index[kk, i] = sim1
#         dis1 = np.sqrt((tm[0] - CN_centroid[0]) ** 2 + (tm[1] - CN_centroid[1]) ** 2)
#         dist[kk, i] = dis1
#         if is_in_ellipse(tm, ell_cn) and flag:
#             sim_index.append(kk)
#             flag = False
#
#     enco_pert = zz1[numofP - 1, :]
#     zz.append(enco_pert)
#
#     distance = np.sqrt((enco_pert[0] - CN_centroid[0]) ** 2 + (enco_pert[1] - CN_centroid[1]) ** 2)
#     dist_last_node2C.append(distance)
#     for i in range(numofP):
#         dist_kick_node2C_AD[i, kk] = np.sqrt((zz1[i, 0] - CN_centroid[0]) ** 2 + (zz1[i, 1] - CN_centroid[1]) ** 2)
# # plt.savefig(save_dir + '/only_sti_AD.png')
# # plt.show()
# np.savetxt(save_dir + '/sim_index_AD_2.txt', sim_index)
# pd.DataFrame(ssim_index).to_csv(save_dir + '/SSIM_AD_2.csv')
# pd.DataFrame(dist).to_csv(save_dir + '/dis_AD_2.csv')


# #
# amplitude = np.arange(0.02, 1.01, 0.02)
#
# me = scipy.io.loadmat(data_dir + '/kick_AD_metastable_sim_avg.mat')
# metastable = np.squeeze(np.array(me['metastable_sim_kick_AD']).T)[:, 0:numofP]

# plot_MateSSIM_for_each_stipara(metastable, ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM-meta_for_amplitude_AD.png')
# plt.show()

# plot_SSIM_for_each_stipara(ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM_for_amplitude_AD.png')
# plt.show()
#
# pert_AD = np.asarray(zz)
# pert_AD_centroid = np.mean(pert_AD, 0)
# dist_last_node2C = np.array(dist_last_node2C)
# dist_full_nodesW[0, :] = dist_last_node2C
# index = [] 
# index = np.argsort(dist_full_nodesW[0])

# print(dist_full_nodesW[0, index])
# print(index)
# np.savetxt(save_dir + '/dist_kick_index_AD_2.txt', index)
#
#
# # plt.figure(figsize=(6, 6))
# XY2 = plot_data_kick_in_latent_XYR(models,
#                               CN[0:100, :],MCI[0:100, :],AD[0:100, :],x_cent,y_cent,2,
#                               kick_AD,label_kick, dist_full_nodesW[0, :],
#                                'x', 1,
#                               batch_size=batch_size,
#                               model_name="vae_mlp",fileName="2309/0-1_0.02_new/kick_AD_10hz_2.png")
# #
# sim_index = []
# fig, ax = plt.subplots()
# encoder, decoder = models
# x1 = CN[0:100, :]
# x2 = MCI[0:100, :]
# x3 = AD[0:100, :]
# z1_mean, _, _ = encoder.predict(x1, batch_size=batch_size)
# z2_mean, _, _ = encoder.predict(x2, batch_size=batch_size)
# z3_mean, _, _ = encoder.predict(x3, batch_size=batch_size)
# plt.scatter(z1_mean[:, 0], z1_mean[:, 1], c='green', alpha=0.5, s=30, linewidths=0)
# plt.scatter(z2_mean[:, 0], z2_mean[:, 1], c='yellow', alpha=0.5, s=30, linewidths=0)
# plt.scatter(z3_mean[:, 0], z3_mean[:, 1], c='red', alpha=0.5, s=30, linewidths=0)
# plt.scatter(x_cent, y_cent, c='blue', marker='D', alpha=1, s=40, linewidths=0)
# confidence = 5.991  # 95%
# make_ellipses(z1_mean, ax, confidence=confidence, color='green', alpha=0.3, eigv=False)
# make_ellipses(z2_mean, ax, confidence=confidence, color='yellow', alpha=0.3, eigv=False)
# make_ellipses(z3_mean, ax, confidence=confidence, color='red', alpha=0.3, eigv=False)
#
# colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
#           'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
#           'cornsilk', 'crimson', 'm', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki',
#           'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen']
# alp = 1
# for sl in range(68):
#     z_mean = Z[:, :, sl]
#     flag = True
#     for i in range(len(z_mean)):
#
#         tm = np.array(z_mean[i, :])
#         if is_in_ellipse(tm, ell_cn) and flag:
#             sim_index.append(sl)
#             flag = False
#
#     if sl >= 34:
#         alp = 0.5
#         cl = colors[sl - 34]
#     else:
#         cl = colors[sl]
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', s=20, c=cl, alpha=alp)
#     plt.plot(z_mean[:, 0], z_mean[:, 1],
#              linewidth=0.5,  
#              linestyle=None, 
#              color='k',  
#              alpha=0.5,
#              markeredgewidth=0.0, 
#              )
# # plt.colorbar()
# plt.xlabel("z[0]")
# plt.ylabel("z[1]")
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.savefig("result/2309/0-1_0.02_new/kick_AD_10hz_2avg.png")
# plt.show()
# np.savetxt(save_dir + '/sim_index_AD_2avg.txt', sim_index)
# #


#mci
kick_MCI = kick_full_MCI[:, 0:numofP, :]
zz = []
dist_full_nodesW = np.zeros((2, numStil))
dist_kick_node2C_MCI = np.zeros((numofP, numStil))
dist_last_node2C = []
# plt.figure(figsize=(6, 6))
sim_index = []
ssim_index = np.zeros((numStil,numofP))
dist = np.zeros((numStil,numofP))
for kk in range(numStil):

    kick = np.transpose(kick_MCI[:, :, kk])
    kick_data = (kick, label_kick)
    zz1 = plot_pert_in_latent(models,
                              kick_data, 'm', 'x', 1,
                              batch_size=batch_size,
                              filename=save_dir + '/only_sti_MCI_1.png')

    # print(len(zz1))
    flag = True
    for i in range(len(zz1)):
        tm = np.array(zz1[i, :])
        zzd = decoder.predict(tm.reshape(1, 2))
        sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
        ssim_index[kk, i] = sim1
        dis1 = np.sqrt((tm[0] - CN_centroid[0]) ** 2 + (tm[1] - CN_centroid[1]) ** 2)
        dist[kk, i] = dis1
        if is_in_ellipse(tm, ell_cn) and flag:
            sim_index.append(kk)
            flag = False

    enco_pert = zz1[numofP - 1, :]
    zz.append(enco_pert)

    distance = np.sqrt((enco_pert[0] - CN_centroid[0]) ** 2 + (enco_pert[1] - CN_centroid[1]) ** 2)
    dist_last_node2C.append(distance)
    for i in range(numofP):
        dist_kick_node2C_MCI[i, kk] = np.sqrt((zz1[i, 0] - CN_centroid[0]) ** 2 + (zz1[i, 1] - CN_centroid[1]) ** 2)
# plt.savefig(save_dir + '/only_sti_MCI.png')
# plt.show()
np.savetxt(save_dir + '/sim_index_MCI.txt', sim_index)
pd.DataFrame(ssim_index).to_csv(save_dir + '/SSIM_MCI.csv')
pd.DataFrame(dist).to_csv(save_dir + '/dis_MCI.csv')


#

me = scipy.io.loadmat(data_dir + '/kick_MCI_metastable_sim.mat')
metastable = np.squeeze(np.array(me['metastable_sim_kick_MCI']).T)[:, 0:numofP]

# plot_MateSSIM_for_each_stipara(metastable, ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM-meta_for_amplitude_AD.png')
# plt.show()
# plot_SSIM_for_each_stipara(ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM_for_amplitude_MCI.png')
# plt.show()

pert_MCI = np.asarray(zz)
pert_MCI_centroid = np.mean(pert_AD, 0)
dist_last_node2C = np.array(dist_last_node2C)
dist_full_nodesW[0, :] = dist_last_node2C
index = []  
index = np.argsort(dist_full_nodesW[0])

print(dist_full_nodesW[0, index])
print(index)
np.savetxt(save_dir + '/dist_kick_index_MCI.txt', index)


# plt.figure(figsize=(6, 6))
XY2 = plot_data_kick_in_latent_XYR(models,
                              CN[0:100, :],MCI[0:100, :],AD[0:100, :],x_cent,y_cent,1,
                              kick_MCI,label_kick, dist_full_nodesW[0, :],
                               'x', 1,
                              batch_size=batch_size,
                              model_name="vae_mlp",fileName=save_dir + "/kick_MCI_10hz_1.png")
print('finish')

#
#

# 
# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_1.mat')
# kick_full_AD1 = mat['FC_sim_kick_AD']
# mat = scipy.io.loadmat(data_dir + '/kick_MCI_FC_sim.mat')
# kick_full_MCI = mat['FC_sim_kick_MCI']


# numofP = kick_full_AD1.shape[1]
# # numofP=30
# label_kick = np.arange(0, numofP)
# numStil=34

#  save ssim
# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_2.mat')
# kick_full_AD2 = mat['FC_sim_kick_AD']
# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_3.mat')
# kick_full_AD3 = mat['FC_sim_kick_AD']
# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_4.mat')
# kick_full_AD2 = mat['FC_sim_kick_AD']
# mat = scipy.io.loadmat(data_dir + '/kick_AD_FC_sim_5.mat')
# kick_full_AD3 = mat['FC_sim_kick_AD']
#
# kick_AD1 = kick_full_AD1[:, 0:numofP, :]
# kick_AD2 = kick_full_AD2[:, 0:numofP, :]
# kick_AD3 = kick_full_AD3[:, 0:numofP, :]
# kick_AD4 = kick_full_AD2[:, 0:numofP, :]
# kick_AD5 = kick_full_AD3[:, 0:numofP, :]
# zz = []
# zza=[]
# dist_full_nodesW = np.zeros((2, numStil))
# dist_kick_node2C_AD = np.zeros((numofP, numStil))
# dist_last_node2C = []
# # plt.figure(figsize=(6, 6))
# sim_index = []
# ssim_index = np.zeros((numStil,numofP))
# dist = np.zeros((numStil,numofP))
# for kk in range(numStil):
#
#     kick = np.transpose(kick_AD1[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz2 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size,filename=save_dir + "/kick_AD_1.png")
#     kick = np.transpose(kick_AD2[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz3 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size,filename=save_dir + "/kick_AD_2.png")
#
#     kick = np.transpose(kick_AD3[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz4 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size,filename=save_dir + "/kick_AD_3.png")
#     kick = np.transpose(kick_AD4[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz5 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size, filename=save_dir + "/kick_AD_4.png")
#
#     kick = np.transpose(kick_AD5[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz6 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size, filename=save_dir + "/kick_AD_5.png")
#     zz1 = (zz2+zz3+zz4+zz5+zz6)/5
#     zza.append(zz1)
#     plt.scatter(zz1[:, 0], zz1[:, 1], marker='x', c=label_kick, alpha=0.3, cmap='viridis')
#     # plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(save_dir + '/kick_AD_navg.png')
#     # print(len(zz1))
#     flag = True
#     for i in range(len(zz1)):
#         tm = np.array(zz1[i, :])
#         zzd = decoder.predict(tm.reshape(1, 2))
#         sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
#         ssim_index[kk, i] = sim1
#         dis1 = np.sqrt((tm[0] - CN_centroid[0]) ** 2 + (tm[1] - CN_centroid[1]) ** 2)
#         dist[kk, i] = dis1
#         if is_in_ellipse(tm, ell_cn) and flag:
#             sim_index.append(kk)
#             flag = False
#
#     enco_pert = zz1[numofP - 1, :]
#     zz.append(enco_pert)
#
#     distance = np.sqrt((enco_pert[0] - CN_centroid[0]) ** 2 + (enco_pert[1] - CN_centroid[1]) ** 2)
#     dist_last_node2C.append(distance)
#     for i in range(numofP):
#         dist_kick_node2C_AD[i, kk] = np.sqrt((zz1[i, 0] - CN_centroid[0]) ** 2 + (zz1[i, 1] - CN_centroid[1]) ** 2)
# plt.savefig(save_dir + '/only_sti_AD_navg.png')
# plt.show()
np.savetxt(save_dir + '/sim_index_AD_navg.txt', sim_index)
pd.DataFrame(ssim_index).to_csv(save_dir + '/SSIM_AD_navg.csv')
pd.DataFrame(dist).to_csv(save_dir + '/dis_AD_navg.csv')


#
amplitude = np.arange(0.02, 1.01, 0.02)


pert_AD = np.asarray(zz)
pert_AD_centroid = np.mean(pert_AD, 0)
dist_last_node2C = np.array(dist_last_node2C)
dist_full_nodesW[0, :] = dist_last_node2C
index = []  
index = np.argsort(dist_full_nodesW[0])

print(dist_full_nodesW[0, index])
print(index)
np.savetxt(save_dir + '/dist_kick_index_AD.txt', index)

def plot_data_kick_in_latent_XYR2(models,
                        data1, data2,data3,xc,yc,l,
                        zz1, label_kick,distance,  symbol, plotsi,
                        batch_size=128,
                        model_name="vae_mnist", fileName="kick"):

    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    fig, ax = plt.subplots()
    encoder, decoder = models
    x1 = data1
    x2 = data2
    x3 = data3
    #os.makedirs(model_name, exist_ok=True)

    filename = os.path.join("result/2309/0-1_0.02_34", fileName)
    # display a 2D plot of the digit classes in the latent space
    z1_mean, _, _ = encoder.predict(x1, batch_size=batch_size)
    z2_mean, _, _ = encoder.predict(x2, batch_size=batch_size)
    z3_mean, _, _ = encoder.predict(x3, batch_size=batch_size)
    #plt.figure(figsize=(10, 10))
    #    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.scatter(z1_mean[:, 0], z1_mean[:, 1], c='green', alpha=0.5, s=30,linewidths=0)
    plt.scatter(z2_mean[:, 0], z2_mean[:, 1], c='yellow', alpha=0.5,s=30,linewidths=0)
    plt.scatter(z3_mean[:, 0], z3_mean[:, 1], c='red', alpha=0.5,s=30,linewidths=0)
    plt.scatter(xc, yc, c='blue',  marker='D',alpha=1, s=40,linewidths=0)


    confidence = 5.991  # 95%
    make_ellipses(z1_mean, ax, confidence=confidence, color='green', alpha=0.3, eigv=False)
    make_ellipses(z2_mean, ax, confidence=confidence, color='yellow', alpha=0.3, eigv=False)
    make_ellipses(z3_mean, ax, confidence=confidence, color='red', alpha=0.3, eigv=False)

    for sl in range(34):
        
        # display a 2D plot of the digit classes in the latent space
        z_mean= zz1[sl]

        z_mean[0, 0] = xc[l]
        z_mean[0, 1] = yc[l]
        # plt.figure(figsize=(12, 10))
        # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        cl=['b','g']
        if plotsi == 1:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', s=20, alpha=1, cmap='viridis')
            # plt.scatter(z_mean[:, 0], z_mean[:, 1], marker='x', s=20, alpha=1, c=cl[sl//34])
            plt.plot(z_mean[:, 0], z_mean[:, 1],
                     linewidth=0.5,  
                     linestyle=None, 
                     color='k',  
                     alpha=1-distance[sl]/np.max(distance)+0.001,
                     markeredgewidth=0.0,  

                     )
    # plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(filename)
    plt.show()


    return z_mean
# plt.figure(figsize=(6, 6))
XY2 = plot_data_kick_in_latent_XYR2(models,
                              CN[0:100, :],MCI[0:100, :],AD[0:100, :],x_cent,y_cent,2,
                              zz1,label_kick, dist_full_nodesW[0, :],
                               'x', 1,
                              batch_size=batch_size,
                              model_name="vae_mlp",fileName="/kick_AD_navg_a.png")

#

#



print('finish')

# #AD
# kick_AD = kick_full_AD[:, 0:numofP, :]
# zz = []
# dist_full_nodesW = np.zeros((2, 68))
# dist_kick_nodesW_AD = np.zeros((numofP, 68))
# dist_min_node2W = []
# # plt.figure(figsize=(6, 6))
# sim_index = []
# ssim_index = []
# for kk in range(68):
#     flag = False
#     kick = np.transpose(kick_AD[:, :, kk])
#     kick_data = (kick, label_kick)
#     zz1 = plot_pert_in_latent(models,
#                               kick_data, 'm', 'x', 1,
#                               batch_size=batch_size,
#                               model_name="vae_mlp")
#
#     # print(len(zz1))
#     for i in range(len(zz1)):
#         tm = np.array(zz1[i, :])
#         # zzd = decoder.predict(tm.reshape(1, 2))
#         if is_in_ellipse(tm, ell_cn):
#             sim_index.append(kk)
#             flag = True
#             break
#
#     if flag:
#         for i in range(len(zz1)):
#             tm = np.array(zz1[i, :])
#             zzd = decoder.predict(tm.reshape(1, 2))
#             sim1 = ssim(de_CN.reshape(68, 68), zzd.reshape(68, 68))
#             ssim_index.append(sim1)
#
#     enco_pert = zz1[numofP - 1, :]
#     zz.append(enco_pert)
#
#     distance = np.sqrt((enco_pert[0] - CN_centroid[0]) ** 2 + (enco_pert[1] - CN_centroid[1]) ** 2)
#     dist_min_node2W.append(distance)
#     for i in range(numofP):
#         dist_kick_nodesW_AD[i, kk] = np.sqrt((zz1[i, 0] - CN_centroid[0]) ** 2 + (zz1[i, 1] - CN_centroid[1]) ** 2)
# plt.savefig(save_dir + '/only_sti_AD.png')
# plt.show()
#
# ssim_index = np.array(ssim_index).reshape(-1, len(zz1))
#
# #
# amplitude = np.arange(10, 151, 10)
#
# me = scipy.io.loadmat(data_dir + '/kick_AD_metastable_sim.mat')
# metastable = np.squeeze(np.array(me['metastable_sim_kick_AD']).T)[:, 0:numofP]
#
# plot_MateSSIM_for_each_stipara(metastable, ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM-meta_for_amplitude_AD.png')
# plt.show()
# plot_SSIM_for_each_stipara(ssim_index, sim_index, amplitude)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
# plt.savefig(save_dir + '/SSIM_for_amplitude_AD.png')
# plt.show()
#
# pert_AD = np.asarray(zz)
# pert_AD_centroid = np.mean(pert_AD, 0)
# dist_min_node2W = np.array(dist_min_node2W)
# dist_full_nodesW[0, :] = dist_min_node2W
# index = []  
# index = np.argsort(dist_full_nodesW[0])
# print(dist_full_nodesW[0, index])
# print(index)

# print(sim_index)
#
# np.savetxt(save_dir + '/dist_kick_index_AD.txt', index)
# np.savetxt(save_dir + '/sim_index_AD.txt', sim_index)
#
# # plt.figure(figsize=(6, 6))
# XY2 = plot_data_kick_in_latent_XYR(models, CN[0:100, :], MCI[0:100, :], AD[0:100, :], x_cent, y_cent, 2, kick_AD,
#                                    label_kick, dist_full_nodesW[0, :], 'x', 1, batch_size=batch_size,
#                                    save_name=save_dir, fileName="kick_AD_10hz.png")
#

