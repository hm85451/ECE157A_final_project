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


def run_algo3(file_path):
    #SVM
    clf = joblib.load('./MLWebApp/algorithms/algorithm3.pkl')
    
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    selected_features = ['PTS', 'TRB', 'AST']
    x = df[selected_features]
    scaler = preprocessing.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    
    y = clf.predict(x_scaled)
    decision = clf.decision_function(x_scaled)
    y_score = clf.score_samples(x_scaled)
    df=df.assign(prediction = y)
    #df=df.assign(Scores = y_score)
    df=df.assign(Scores = decision)
    df = df.sort_values(by=['Scores'], ascending= True)

    three_outliers = df[:3]
    names = three_outliers['Player']
    names = names.values.tolist()

    df = df.reset_index(drop=True)
    data_score = df[['Player','Scores']]
    data_score =data_score.drop_duplicates(subset=['Player'],keep='first')
    data_score.to_csv('Elliptical_Envelope_Scores.csv', index=False,)  
    
    #plot the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    df = df.reset_index(drop=True)
    for ind in df.index:
        x1=df[selected_features[0]][ind]
        x2=df[selected_features[1]][ind]
        x3=df[selected_features[2]][ind]

        if df['prediction'][ind] < 0:
            if ind < 3:
                ax.scatter(x1, x2, x3, marker = '^', color='g', depthshade = False)
            else:
                ax.scatter(x1, x2, x3, color='r', depthshade = False)
        else:
            ax.scatter(x1, x2, x3, color='b',depthshade = False)

    #plt.show
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_zlabel(selected_features[2])
    plt.title('Inliers and Outliers in 3D')
    with io.StringIO() as stringbuffer:
        fig.savefig(stringbuffer,format='svg')
        svgstring = stringbuffer.getvalue()
    return svgstring