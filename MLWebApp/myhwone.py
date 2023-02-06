import joblib
import mpld3
import numpy as np
import csv

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_algo1(file_path):
    # Load file_path instead of unknowns.csv
# Do the same processing you did to
# get your unknowns.csv results from your model.
# Get the figure object Matplotlib will be working with:
    clf = joblib.load('./MLWebApp/algorithms/algorithm1.pkl')
    temp = genfromtxt(file_path, delimiter=',')
    unknown = np.delete(temp,0,0)
    positive = []
    negative = []

    f = open('scores.csv', 'w',newline='')

    writer = csv.writer(f)

    writer.writerow(['Outcomes'])

    for i in range(len(unknown)):
        y = clf.predict([unknown[i]])
        if y > 0:
            positive.append(unknown[i])
        else:
            negative.append(unknown[i])
        writer.writerow(y)
    f.close()
    
    positive = np.array(positive)
    negative = np.array(negative)


    fig = plt.figure()
    
# Scatterplot your results, as per
# any of your scatterplots from homework 1.
# Make sure to use plt.xlabel(), plt.ylabel()
# and plt.title() to give clear labeling!
    plt.scatter(positive[:,1:2], positive[:,2:3], alpha=0.5)
    plt.scatter(negative[:,1:2], negative[:,2:3], alpha=0.5)
    plt.ylabel('Blood Pressure')
    plt.xlabel('Glucose')
    plt.title("misclassified vs classified")
    return mpld3.fig_to_html(fig)
