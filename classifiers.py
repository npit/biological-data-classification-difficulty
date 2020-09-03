import os
import numpy as np
import pandas as pd
from shutil import copyfile
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import time

def gather_and_save_all_accuracies(df, params, classifiers, num_folds, results, results_path,
    metafeat, metafeatures_columns):
    filename ,representation, instance_frac, feature_frac = params
    # all_accuracy_scores = []
    # get accuracy from a range of classifiers via cross-validation
    for classifier_name, classifier_func in classifiers.items():
        # generate train / validation indexes
        splitter = StratifiedKFold(n_splits=num_folds)

        time_start = time.time()
        for fold_index, (train_index, val_index) in enumerate(splitter.split(df.values, df.values[:, -1])):
            # make an ID for recoverable progress
            current_id = f"{classifier_name}_fold{fold_index}_repr{representation}_ifrac{instance_frac}_ffrac{feature_frac}"
            if current_id in results["exp_id"].values.tolist():
                print(f"Skipping experiment: {current_id} as it's already completed.")
                # accuracy = results[results["exp_id"]==current_id]["accuracy"]
                # all_accuracy_scores.append(accuracy)
                continue
            # print(f"Running classifier {classifier_name} on fold {fold_index+1}/{num_folds} on {len(train_index)} train and {len(val_index)} val data")
            # get data and labels of the current fold
            train_data = df.values[train_index, :-1]
            train_labels = df.values[train_index,-1]
            val_data = df.values[val_index, :-1]
            val_labels = df.values[val_index,-1]
            # print(f"Data shapes: {train_data.shape}, {val_data.shape}, label shapes: {train_labels.shape}, {val_labels.shape}, num_classes: {len(set(val_labels))}")
            # run the classifier
            accuracy = classifier_func(train_data, train_labels, val_data, val_labels)
            # columns are representation inst_frac feat_frac classifier fold
            row_data = [current_id, representation, filename, instance_frac, feature_frac, classifier_name, fold_index, accuracy]

            # add metafeatures
            for metafeat_name in metafeatures_columns:
                row_data.append(metafeat[metafeat_name])

            results = results.append(pd.Series(row_data, index=results.columns ), ignore_index=True)
            # make a backup of the results file before copying
            if os.path.exists(results_path):
                copyfile(results_path, results_path + ".backup")
            results.to_csv(results_path, index=None)
            # all_accuracy_scores.append(accuracy)
        time_end = time.time()
        el = time_end - time_start
        print(f"{num_folds}-folds for {classifier_name} : elapsed {el/60} minutes / {el} sec")
    return results



def decision_tree(x_train, y_train, x_test, y_test):
    # y=a[a.columns[-1]]
    # X = a[a.columns[:-1]]
    # x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#,random_state=42)
    #x_train = preprocessing.scale(x_train) #standardization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train= min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictionsDT=model.predict(x_test)
    a4=accuracy_score(y_test,predictionsDT)
    return(a4)

def logistic_regression(x_train, y_train, x_test, y_test):
    # y=a[a.columns[-1]]
    # X = a[a.columns[:-1]]
    # x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#,random_state=42)
    #x_train = preprocessing.scale(x_train) #standardization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train= min_max_scaler.fit_transform(x_train)
    # also apply scaling to the test data
    x_test= min_max_scaler.transform(x_test)
    model=LogisticRegression(max_iter=1000)
    model.fit(x_train,y_train)
    predictionsLR=model.predict(x_test)
    a1=accuracy_score(y_test,predictionsLR)
    return(a1)

def SVM(x_train, y_train, x_test, y_test):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(x_train, y_train)
    predictionsSVC= svclassifier.predict(x_test)
    a2=accuracy_score(y_test,predictionsSVC)
    #print (a2)
    return(a2)

def KNN(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)# 5 is our choice
    knn.fit(x_train, y_train)
    predictionsKNN = knn.predict(x_test)
    a3=accuracy_score(y_test,predictionsKNN)
    #print (a3)
    return(a3)

def NN(x_train, y_train, x_test, y_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    data_dim = x_train.shape[-1]
    num_classes = len(set(y_test))
    
    # Dependencies
    # Neural network
    model = Sequential()
    model.add(Dense(512, input_dim=data_dim, activation='relu'))# we are using two hidden layers of 16 and 12 dimension.
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # train and predict
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ytrain_onehot = to_categorical(y_train)
    
    # print(ytrain_onehot)
    # print(y_train)

    model.fit(x_train, ytrain_onehot, epochs=25, batch_size=64, verbose=0)
    y_pred = model.predict(x_test)
    predicted_labels = np.argmax(y_pred, axis=1) 

    #Converting predictions to label
    # print('Accuracy is:', a*100, " -- ", accuracy_score(predicted_labels, y_test))
    return accuracy_score(predicted_labels,y_test)
