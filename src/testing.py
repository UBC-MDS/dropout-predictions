# Author: Andy Wang, Ranjit Sundaramurthi, Caesar
# Date: 2022-11-25

"""
A script that test and plot the result for best model and store it under the results/ folder.

Usage: src/test_all_model.py --test=<test> --out_dir=<out_dir>
 
Options:
--test=<test>        Input path for the test dataset
--out_dir=<out_dir>   Path to directory where the serialized model should be written

"""

# Example:
# python test_all_model.py --test="../data/processed/test.csv" --out_dir="../results/"

# import
from docopt import docopt
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

opt = docopt(__doc__) # This would parse into dictionary in python

def main(test, out_dir):
    # get test data
    test_df = pd.read_csv(test)
    X_test, y_test = test_df.drop(columns=["Target"]), test_df["Target"]

    # get the best model 
    # best model 1: logistic regression
    models = {'logisticRegression': '',
              'NaiveBayes': '',
              'RandomForestClassifier': ''}
    for model_name in models:
        cur_file = open(out_dir + 'model_' + model_name, 'rb')
        cur_final_model = pickle.load(cur_file)
        models[model_name] = cur_final_model
    print(models)

    
    # plot PR curve
    for model_name in models:
        precision, recall, thresholds = precision_recall_curve(
                                    y_test, models[model_name].predict_proba(X_test)[:, 1])
        plot_x_y(precision, recall, model_name)

    plt.title("PR curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="lower left")
    plt.savefig(out_dir + 'PR_curve.png')

    # clean plot
    plt.clf()


    for model_name in models:
        if model_name != 'logisticRegression':
            fpr, tpr, thresholds = roc_curve(y_test, models[model_name].predict_proba(X_test)[:, 1])
        else:
            fpr, tpr, thresholds = roc_curve(y_test, models[model_name].decision_function(X_test))
        plot_x_y(fpr, tpr, model_name)
    plt.title("ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.savefig(out_dir + 'ROC_curve.png')

    # clean plot
    plt.clf()
    
    
    # # get the score
    # pred = get_result(final_model_log, X_test, y_test)
    # result_log = open(out_dir + '/result_log', 'wb')
    # pickle.dump(pred, result_log)
    # result_log.close()

def plot_x_y(x_data, y_data, label):
    plt.plot(x_data, y_data, label=label)
    return plt

def get_result(final_model, test_x, test_y):
    return classification_report(test_y, final_model.predict(test_x), target_names=["Graduate", "Drop"])


if __name__ == "__main__":
    main(opt["--test"], opt["--out_dir"])