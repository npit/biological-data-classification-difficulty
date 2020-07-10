# imports
import sys
import pandas as pd
import arff
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.feature_selection import mutual_info_classif
from info_gain import info_gain
from sklearn.svm import SVC
from sklearn import linear_model
from random import randint
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from classifiers import *

# NNs

# google-collab settings
# installation of dependencies for execution
# !pip install -q liac-arff pymfe pandas sklearn info_gain numpy

"""# New Section"""

def import_data_from_colab():
    from google.colab import files


METAFEATURES_COLUMNS = ['nr_instances', 'nr_features', 'nr_missing_values', 'mean_kurtosis', 'mean_skewness', 'mean', 'Info_gain', 'Inf_gain_ratio']
def read_arff_data(filepath):
    with open (filepath) as f:
        decoder=arff.ArffDecoder()
        datadictionary=decoder.decode(f,encode_nominal=True,return_type=arff.LOD)
        data=datadictionary['data']
        full_df = pd.DataFrame(data)
    return full_df

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

# data saveloading parameters
#############################
# do not change this
base_google_drive_path =  "/content/drive/My Drive/"
# change this with where Marina's data are (folders with arffs)
google_drive_folder_with_representations_data = "Colab Notebooks/marina_biological_data"
get_input_data_from_drive = False
# for local runs use this
folder_with_representations_data = "Combined Representations"
#############################
# experiment parameters
# #####################
only_run_reprs = ["Zcurve"] # repr. names here to limit run to these represenations
min_num_features = 100
num_folds = 3
instance_fractions = [0.5, 0.8, 1.0]
feature_fractions = [0.5, 0.8, 1.0]
classifiers = {"NN": NN, "KNN": KNN, "SVM": SVM, "LR": logistic_regression, "DT": decision_tree}
results_path_file = "results.csv"
# ####################

# input data
if get_input_data_from_drive:
    run_data_folder = base_google_drive_path + google_drive_folder_with_representations_data
    results_path = base_google_drive_path + "/" + google_drive_folder_with_representations_data +  "/" + results_path_file 
    if not os.path.exists(run_data_folder):
        from google.colab import drive
        drive.mount('/content/drive')
else:
    run_data_folder = folder_with_representations_data
    results_path = folder_with_representations_data + "/" + results_path_file 

print("Will load data from:", run_data_folder)
print("Will save results data to", results_path)
print("Working in directory:", os.getcwd())
# output data: make or load the results file
if os.path.exists(results_path):
    results = pd.read_csv(results_path)
else:
    results = pd.DataFrame(columns="exp_id representation filename inst_frac feat_frac classifier fold accuracy".split() + METAFEATURES_COLUMNS)


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
    print("Representation",representation)
    # for each dataset
    for filename in os.listdir(representation_folder):
        # arff filepath
        filepath = os.path.join(representation_folder, filename)
        full_df = read_arff_data(filepath)
        import ipdb; ipdb.set_trace()
        full_features = list(full_df.columns)[:-1]
        label_column = full_df.columns[-1]

        print("Running file ", filepath, "which has full data:", full_df.values.shape)
        # keep track of metafeatures and average accuracies
        metafeatures, dataset_accuracies, full_paths = [], [], []

        # for each fraction of instances
        for instance_frac in instance_fractions:
            # select a different random percentage of instances 
            instance_fractioned_df = frac(full_df, instance_frac)
            print(f"Data fractioned by {instance_frac} instances:", instance_fractioned_df.values.shape)

            # for each fraction of features
            for feature_frac in instance_fractions:
                subset_size = round(feature_frac * len(full_features))
                # select a different random percentage of features 
                features_subset = random.sample(full_features, subset_size)
                selected_features = features_subset + [label_column]
                df = instance_fractioned_df.loc[:, selected_features]
                config_counter += 1
                print(f"Configuration {config_counter} / {num_all_configurations} -- Feature fraction: {feature_frac} = {subset_size}, resulting in data matrix: {df.values.shape}")

                # delete all-nan columns, replace partial nans with zero
                df = df.dropna(1, how="all")
                df = df.fillna(0.0)

                if df.isna().values.any():
                    print("NANs exist!")
                    # debug
                    import ipdb; ipdb.set_trace()

                # average performance across all classifiers and folds
                metafeat = extract_metafeature(df)
                metafeatures.append(metafeat)
                accuracies = gather_and_save_all_accuracies(df, (filename, representation, instance_frac, feature_frac), classifiers, num_folds, results, results_path, metafeat, METAFEATURES_COLUMNS)
                # dataset metafeatures for the current instance / features fraction 
                dataset_accuracies.append(accuracies)
                full_paths.append(filepath)

#ski-learn-imputation
#+1 nn mikro 
#+1 decision tree-done
#plires run -done
#datasets me (2-100) classes