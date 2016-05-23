#!/usr/bin/env python

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import time
import numpy as np

fig=plt.figure(1)
ax=fig.add_subplot(111)

def plotError(train, validation, test, path, name):
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.figure(1)
    lines = plt.plot(train[0], train[1], validation[0], validation[1], \
        test[0], test[1])
    plt.setp(lines[0], color='g', linewidth=2.0, label='Training')
    plt.setp(lines[1], color='b', linewidth=2.0, label='Validation')
    plt.setp(lines[2], color='r', linewidth=2.0, label='Testing')

    g_patch = mpatches.Patch(color='g', label='Training Loss Curve')
    b_patch = mpatches.Patch(color='b', label='Validation Loss Curve')
    r_patch = mpatches.Patch(color='r', label='Testing Loss Curve')

    plt.legend(handles=[g_patch, b_patch, r_patch]) #, bbox_to_anchor=(.95, 0.25))

    plt.title('Error / Epoch')
    plt.grid(True)
    plt.savefig(path + '/' +  name)

def livePlotTrError(train, val=None, test=None):
    line,=ax.plot(train[0],train[1],'ko-')
    line.set_data(train[0],train[1])  
    plt.pause(1)