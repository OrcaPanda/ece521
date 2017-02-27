###############################################################
# Part 2.4 Visualization
###############################################################

import tensorflow as tf
import numpy as np
import random
import sys
import matplotlib
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    W_d_25 = pickle.load(open("dropout1","rb"))
    W_d_50 = pickle.load(open("dropout2", "rb"))
    W_d_75 = pickle.load(open("dropout3", "rb"))
    W_d_100 = pickle.load(open("dropout4", "rb"))
    W_nd_25 = pickle.load(open("no_dropout1","rb"))
    W_nd_50 = pickle.load(open("no_dropout2", "rb"))
    W_nd_75 = pickle.load(open("no_dropout3", "rb"))
    W_nd_100 = pickle.load(open("no_dropout4", "rb"))

    IMAGE = np.zeros((40*28, 25*28))
    for y in list(range(40)):
       for x in list(range(25)):
           for y1 in list(range(28)):
               for x1 in list(range(28)):
                   IMAGE[y*28+y1,x*28+x1] = W_d_100[y1*28+x1,y*25+x]


    vmin = np.amin(IMAGE)
    vmax = np.amax(IMAGE)
    plt.imshow(IMAGE, cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    plt.show()

    #for x in list(range(100)):

