###############################################################
# Part 2.3.1 Different number of hidden units
###############################################################

import tensorflow as tf
import numpy as np
import random
import sys

import matplotlib
import matplotlib.pyplot as plt

import pickle

stopping_point = 150

point = np.array([0.25, 0.5, 0.75, 1.0])
point = np.floor( point * stopping_point / 5.0 ) * 5
point = point.astype(int)
point = point - 1

print(point)
for val in point:

    W_nd = pickle.load(open("q2_4_dropout/no_dropout"+str(val), "rb"))
    W_d = pickle.load(open("q2_4_dropout/dropout"+str(val), "rb"))

    boarder_width = 2
    IMAGE = np.zeros((10 * 28 + boarder_width * 10, 10 * 28 + boarder_width * 10))
    IMAGE[:, :] = np.amin(W_nd)
    for y in list(range(10)):
        for x in list(range(10)):
            for y1 in list(range(28)):
                for x1 in list(range(28)):
                    IMAGE[(y ) * (28 + boarder_width) + y1, (x )* (28 + boarder_width) + x1] = W_d[y1 * 28 + x1, y * 10 + x]

    vmin = np.amin(IMAGE)
    vmax = np.amax(IMAGE)
    plt.imshow(IMAGE, cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    plt.savefig('drop_out_visualization' + str(val) + '.png')
    plt.show()

    IMAGE = np.zeros((10 * 28+ boarder_width*10, 10 * 28+ boarder_width*10))
    IMAGE[:,:] = np.amin(W_nd)
    for y in list(range(10)):
        for x in list(range(10)):
            for y1 in list(range(28)):
                for x1 in list(range(28)):
                    IMAGE[(y) * (28+boarder_width) + y1, (x) * (28 + boarder_width) + x1] = W_nd[y1 * 28 + x1, y * 25 + x]
    vmin = np.amin(IMAGE)
    vmax = np.amax(IMAGE)
    plt.imshow(IMAGE, cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    plt.savefig('no_drop_out_visualization' + str(val) + '.png')
    plt.show()