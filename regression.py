import numpy as np
import pandas as pd
import os
import shutil
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# def decision_tree(x_train, y_train, x_test, y_test):
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_train= min_max_scaler.fit_transform(x_train)
#     x_test = min_max_scaler.transform(x_test)
#     model = DecisionTreeRegressor()
#     model.fit(x_train, y_train)
#     preds = model.predict(x_test)
#     return mean_squared_error(y_test,predictionsDT)

# def SVM(x_train, y_train, x_test, y_test):
#     regressor = SVR(kernel='rbf')
#     svclassifier.fit(x_train, y_train)
#     predictionsSVC= svclassifier.predict(x_test)
#     a2=accuracy_score(y_test,predictionsSVC)
#     #print (a2)
#     return(a2)

# def KNN(x_train, y_train, x_test, y_test):
#     knn = KNeighborsRegressor(n_neighbors=2)# 5 is our choice
#     knn.fit(x_train, y_train)
#     predictionsKNN = knn.predict(x_test)
#     a3=accuracy_score(y_test,predictionsKNN)
#     return(a3)

# def NN(x_train, y_train, x_test, y_test):
#    # https://stackoverflow.com/questions/16879928/neural-networks-regression-using-pybrain
#    pass

def run_regression(regressor, x_train, y_train, x_test, y_test, mmx_scale=False):
    if mmx_scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train= min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.fit_transform(x_test)
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    return mean_squared_error(y_test, preds)

def get_data_and_labels_from_raw_inputs(input_data):
    metafeatures_columns = "nr_instances	nr_features	nr_missing_values	mean_kurtosis	mean_skewness	mean	Info_gain	Inf_gain_ratio".split()
    accuracy_column = "accuracy"

    data = input_data[metafeatures_columns].values
    labels = input_data[accuracy_column].values
    return data, labels

def main(input_file, num_folds=2, seed=1337, output_path="regression_results.csv"):

    # opts
    regressors = {
        "svr" : lambda : SVR(),
        "dtr": lambda : DecisionTreeRegressor(),
        "knn": lambda : KNeighborsRegressor(n_neighbors=2)
    }

    # load data
    # Expected csv format:
    # exp_id	representation	filename	inst_frac	feat_frac	classifier	fold
    # accuracy	nr_instances	nr_features	nr_missing_values	mean_kurtosis	mean_skewness
    # mean	Info_gain	Inf_gain_ratio

    input_data = pd.read_csv(input_file)
    print("Read raw input data with shape:", input_data.shape)


    # Given the input data, potentially aggregate some accuracy values (e.g. over all folds)
    data, labels = get_data_and_labels_from_raw_inputs(input_data)
    print(f"Running regressions on {len(data)} data/label instances.")

    print("Run regression in terms of a single representation? GG clarify!")

    output_cols = "id score".split()
    results = pd.DataFrame(columns=output_cols)

    # cross-val
    splitter = KFold(num_folds, shuffle=True, random_state=seed)
    # iterate over folds
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]
        # iterate over regressors
        for regressor_name, model_func in regressors.items():
            # run the regression
            print(f"Running fold {fold_idx+1}/{num_folds} : {regressor_name}")
            regressor = model_func()
            score = run_regression(regressor, x_train, y_train, x_test, y_test, mmx_scale=(regressor_name == "dtr"))
            # get a run id, store results
            run_id = f"regressor_{regressor_name}_fold_{fold_idx}"
            results = results.append({"id": run_id, "score": score}, ignore_index=True)
            # backup a copy
            shutil.copyfile(output_path, output_path + " .backup.csv")
            results.to_csv(output_path, index=None)



if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        print("Need an input csv file")