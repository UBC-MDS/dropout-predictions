# Author: Andy Wang
# Date: 2022-11-23

"""
A script that test and plot the result for best model and store it under the result/ folder.

Usage: src/testing.py --test=<test> --out_dir=<out_dir>
 
Options:
--test=<test>        Input path for the test dataset
--out_dir=<out_dir>   Path to directory where the serialized model should be written

"""

# Example:
# python testing.py --test="../data/processed/test.csv" --out_dir="../result/"

# import
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os
import pickle
import pandas as pd
from scipy.stats import lognorm, loguniform, randint
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

opt = docopt(__doc__) # This would parse into dictionary in python

def main(test, out_dir):
    # get test data
    test_df = pd.read_csv(test)
    X_test, y_test = test_df.drop(columns=["Target"]), test_df["Target"]

    # get the best model 
    file1 = open('../result/model_try', 'rb')
    final_model_log = pickle.load(file1)
    
    plot_model(final_model_log, X_test, y_test, 'Logistic Regression')
    plt.savefig('../result/log-reg.png')

    # get the score
    pred = get_result(final_model_log, X_test, y_test)
    result_log = open(out_dir + '/result_log', 'wb')
    pickle.dump(pred, result_log)
    result_log.close()
    
def plot_model(final_model, test_x, test_y, name):
    precision, recall, thresholds = precision_recall_curve(
    test_y, final_model.predict_proba(test_x)[:, 1])

    plot = plt.plot(precision, recall, label=name)
    return plot

def get_result(final_model, test_x, test_y):
    return classification_report(test_y, final_model.predict(test_x), target_names=["Graduate", "Drop"])


if __name__ == "__main__":
    main(opt["--test"], opt["--out_dir"])