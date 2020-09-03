# imports
import sys
import pandas as pd
import numpy as np
import math
import random
from statistics import *
from pymfe.mfe import MFE
from statistics import *
from scipy.stats import skew
from scipy.stats import kurtosis
from pandas import DataFrame
import os
from util import read_arff_data

from sklearn.feature_selection import mutual_info_classif
from info_gain import info_gain
from sklearn.svm import SVC
from sklearn import linear_model
from random import randint
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from classifiers import *
from argparse import ArgumentParser

# NNs

# google-collab settings
# installation of dependencies for execution
# !pip install -q liac-arff pymfe pandas sklearn info_gain numpy

"""# New Section"""

def import_data_from_colab():
    from google.colab import files


METAFEATURES_COLUMNS = ['nr_instances', 'nr_features', 'nr_missing_values', 'mean_kurtosis', 'mean_skewness', 'mean', 'Info_gain', 'Inf_gain_ratio']

def extract_metafeature(a):
    y=a[a.columns[-1]]
    X =a[a.columns[:-1]]
    return {
     #simple
    'nr_instances': len(a),
    'nr_features': len(a.columns),
    'nr_missing_values': a.isnull().sum().sum(),
    'mean_kurtosis': mean(a.kurtosis()),
    'mean_skewness': mean(a.skew()),
    'mean': mean(a.mean()),
    'Info_gain': info_gain.info_gain(X,y),
    'Inf_gain_ratio': info_gain.info_gain_ratio(X,y)
    }

def frac(dataframe, fraction):
    """Returns fraction of data"""
    return dataframe.sample(frac=fraction)

def random_gen():
    """generates random number"""
    return random.randint(0,1)


def main():

    parser = ArgumentParser()
    parser.add_argument("-rep_folder", help="Folder with representation folders.", default="representations")
    parser.add_argument("-input_type", help="Specify input from google drive or locally.", choices=["drive", "local"], default="local")
    parser.add_argument("-only_rep", help="Limit to only the specified representation.", default=False)
 
    args = parser.parse_args()

    # data saveloading parameters
    #############################
    # change this with where Marina's data are (folders with arffs)
    get_input_data_from_drive = args.input_type == "drive"
    folder_with_representations_data = os.path.join(os.getcwd(), args.rep_folder)
    #############################

    # experiment parameters
    # #####################
    only_run_reprs = args.only_rep  # repr. names here to limit run to these represenations
    min_num_features = 100
    num_folds = 3
    instance_fractions = [0.5, 0.8, 1.0]
    feature_fractions = [0.5, 0.8, 1.0]
    classifiers = {"NN": NN, "KNN": KNN, "SVM": SVM, "LR": logistic_regression, "DT": decision_tree}
    results_path_file = "results.csv"
    # ####################

    # input data
    if get_input_data_from_drive:
        base_google_drive_path =  "/content/drive/My Drive/"
        run_data_folder = base_google_drive_path + folder_with_representations_data
        results_path = base_google_drive_path + "/" + folder_with_representations_data +  "/" + results_path_file 
        if not os.path.exists(run_data_folder):
            from google.colab import drive
            drive.mount('/content/drive')
    else:
        run_data_folder = folder_with_representations_data
        results_path = run_data_folder + "/" + results_path_file
        if not os.path.exists(run_data_folder):
            print("Can't find represenations folder:", run_data_folder)
            exit()

    print("Will load data from:", run_data_folder)
    print("Will save results data to", results_path)
    print("Working in directory:", os.getcwd())
    # output data: make or load the results file
    if os.path.exists(results_path):
        print("Continuing from existing results:", results_path)
        results = pd.read_csv(results_path)
    else:
        print("Creating brand-new results:", results_path)
        results = pd.DataFrame(columns="exp_id representation filename inst_frac feat_frac classifier fold accuracy".split() + METAFEATURES_COLUMNS)
        try:
            results.to_csv(results_path)
        except:
            print("Can't make the results file path:", run_data_folder)
            exit()


    if not only_run_reprs:
        representation_folders = [x for x in os.listdir(run_data_folder) if os.path.isdir(os.path.join(run_data_folder, x)) and not x.startswith(".")]
    else:
        representation_folders = [x for x in only_run_reprs if os.path.isdir(os.path.join(run_data_folder, x)) and not x.startswith(".")]

    print("Representations:", representation_folders)

    # 
    num_all_datasets = 271
    num_all_configurations = num_all_datasets * len(instance_fractions) * len(feature_fractions)
    num_all_classifications = num_all_configurations * len(classifiers) * num_folds
    print("All configurations:", num_all_configurations)
    print("All classifications:", num_all_classifications)

    config_counter = 0
    # for each representation
    for representation in representation_folders:
        representation_folder = os.path.join(run_data_folder, representation)
        print("Starting representation: ",representation)
        # for each dataset
        for filename in os.listdir(representation_folder):
            # arff filepath
            filepath = os.path.join(representation_folder, filename)
            full_df = read_arff_data(filepath)
            full_features = list(full_df.columns)[:-1]
            label_column = full_df.columns[-1]

            # print("Running file ", filepath, "which has full data:", full_df.values.shape)

            # for each fraction of instances
            for instance_frac in instance_fractions:
                # select a different random percentage of instances 
                instance_fractioned_df = frac(full_df, instance_frac)
                # print(f"Data fractioned by {instance_frac} instances:", instance_fractioned_df.values.shape)

                # for each fraction of features
                for feature_frac in instance_fractions:
                    subset_size = round(feature_frac * len(full_features))
                    # select a different random percentage of features 
                    features_subset = random.sample(full_features, subset_size)
                    selected_features = features_subset + [label_column]
                    df = instance_fractioned_df.loc[:, selected_features]
                    config_counter += 1
                    msg = f"{representation} {filename} fulldata: {full_df.values.shape} | {config_counter} / {num_all_configurations} "
                    msg += f"-- ffrac: {feature_frac}, ifrac: {instance_frac} data shape: {df.values.shape}"
                    print(msg)

                    # delete all-nan columns, replace partial nans with zero
                    df = df.dropna(1, how="all")
                    df = df.fillna(0.0)

                    if df.isna().values.any():
                        print("NANs exist!")
                        # debug

                    # average performance across all classifiers and folds
                    metafeat = extract_metafeature(df)
                    results = gather_and_save_all_accuracies(df, (filename, representation, instance_frac, feature_frac), classifiers, num_folds, results, results_path, metafeat, METAFEATURES_COLUMNS)
                    # dataset metafeatures for the current instance / features fraction 
                    del df

    #ski-learn-imputation
    #+1 nn mikro 
    #+1 decision tree-done
    #plires run -done
    #datasets me (2-100) classes

if __name__ == "__main__":
    main()