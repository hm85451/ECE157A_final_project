import joblib
import mpld3
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')

def run_algo2(file_path):
    clf = joblib.load('./MLWebApp/algorithms/algorithm2.pkl')
    temp = genfromtxt(file_path, delimiter=',')
    unknown = np.delete(temp,0,0)
    scaler = StandardScaler()
    unknown = scaler.fit_transform(unknown)
    plot = []
    
    f = open('scores2.csv', 'w',newline='')

    writer = csv.writer(f)

    writer.writerow(['Outcomes'])

    for i in range(len(unknown)):
        y = clf.predict([unknown[i]])
        y_int = [round(y[0],0)]
        plot.append([y_int,y])
        writer.writerow(y)
    f.close()
    
    plot = np.array(plot)
    fig = plt.figure()
    plt.scatter(plot[:,0:1], plot[:,1:2])
    plt.ylabel('Model Prediction')
    plt.xlabel('Ground Truth')
    plt.title("Wine Quality Regression Results")
    return mpld3.fig_to_html(fig)