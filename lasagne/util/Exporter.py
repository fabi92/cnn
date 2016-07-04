#!/usr/bin/env python

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import lasagne
import numpy as np

def plotError(train, validation, test, name):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.figure(1)
    lines = plt.plot(train[0], train[1], validation[0], validation[1], \
        test[0], test[1])
    plt.setp(lines[0], color='g', linewidth=2.0, label='Training')
    plt.setp(lines[1], color='b', linewidth=2.0, label='Validation')
    plt.setp(lines[2], color='r', linewidth=2.0, label='Testing')

    g_patch = mpatches.Patch(color='g', label='Training Loss Curve')
    b_patch = mpatches.Patch(color='b', label='Validation Loss Curve')
    r_patch = mpatches.Patch(color='r', label='Testing Loss Curve')

    plt.legend() #, bbox_to_anchor=(.95, 0.25))

    plt.title('Error / Loss')
    plt.grid(True)
    plt.savefig(name)

def csvExport(trainRes, valRes, testRes, exportTxtFile):
    header = "Epoch;Train MSE;Validation MSE;Test MSE;Train MAE;Validation MAE;Test MAE;Train R2;Validation R2;Test R2;\n"
    body = ""
    for idx in range(len(trainRes[0])):
        body = body + str(idx+1) \
                + ";" + str(trainRes[1][idx]) + ";" + str(valRes[1][idx]) + ";" + str(testRes[1][idx]) \
                + ";" + str(trainRes[2][idx]) + ";" + str(valRes[2][idx]) + ";" + str(testRes[2][idx]) \
                + ";" + str(trainRes[3][idx]) + ";" + str(valRes[3][idx]) + ";" + str(testRes[3][idx]) + "\n"

    with open(exportTxtFile, "w") as t_file:
        content = header + body
        t_file.write(content)

def csvExportHeader(exportTxtFile):
    header = "Epoch;Train MSE;Validation MSE;Test MSE;Train MAE;Validation MAE;Test MAE;Train R2;Validation R2;Test R2;\n"
    with open(exportTxtFile, "w") as t_file:
        t_file.write(header)

def csvExportEpoch(epoch, trainRes, valRes, testRes, exportTxtFile):
    body = str(epoch) \
            + ";" + str(trainRes[0]) + ";" + str(valRes[0]) + ";" + str(testRes[0]) \
            + ";" + str(trainRes[1]) + ";" + str(valRes[1]) + ";" + str(testRes[1]) \
            + ";" + str(trainRes[2]) + ";" + str(valRes[2]) + ";" + str(testRes[2]) + "\n"

    with open(exportTxtFile, "a") as t_file:
        t_file.write(body)

def exportModel(exportModelFile, cnn):
    np.savez(exportModelFile, *lasagne.layers.get_all_param_values(cnn))