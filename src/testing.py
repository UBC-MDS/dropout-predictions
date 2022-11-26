# Author: Andy Wang, Ranjit Sundaramurthi, Caesar
# Date: 2022-11-25

"""
A script that test and plot the result for best model and store it under the results/ folder.

Usage: src/testing.py --test=<test> --out_dir=<out_dir>
 
Options:
--test=<test>        Input path for the test dataset
--out_dir=<out_dir>   Path to directory where the serialized model should be written

"""

# Example:
# python testing.py --test="../data/processed/test.csv" --out_dir="../results/"

# import
from docopt import docopt
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

opt = docopt(__doc__) # This would parse into dictionary in python

def main(test, out_dir):
    '''
    read all the stored model generated from training.py
    Predict the y_test data
    Generate confusion matrix, ROC curve, PR curve
    
    Parameters
    ----------
    test : str
        testing data path

    out_dir : str
        output directory path
     
    Returns
    -------
    store 2 curve plot (ROC, PR)
    store 3 confusion matrix png

    Examples
    --------
    >>> main(opt["--train"], opt["--scoring_metrics"], opt["--out_dir"])
    '''

    # get test data
    test_df = pd.read_csv(test)
    X_test, y_test = test_df.drop(columns=["Target"]), test_df["Target"]

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

    # plotting ROC curve
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
    
    # plotting confusion matrix 
    for model_name in models:
        y_test = test_df["Target"]
        # report = get_result(models[model_name], X_test, y_test)
        # df = pd.DataFrame(report).transpose()
        # print(df)
        y_predict = models[model_name].predict(X_test)
        y_predict = pd.DataFrame(y_predict, columns=['Target'])
        y_test = pd.DataFrame(y_test, columns=['Target'])

        target_dict = {0: 'Graduate', 1: 'Dropout'}

        y_predict = y_predict.replace({'Target': target_dict})
        y_predict = y_predict.Target.tolist()
        y_test = y_test.replace({'Target': target_dict})
        y_test = y_test.Target.tolist()

        cm_df = pd.DataFrame(confusion_matrix(y_test, y_predict),['Graduate', 'Dropout'],['Graduate', 'Dropout'])                      
        plt.figure(figsize=(10,6))  
        sns.heatmap(cm_df, annot=True, cmap="crest", fmt='d').set(title='Confusion Matrix ' + model_name)

        plt.savefig(out_dir + 'Confusion_Matrix_'+ model_name+'.png')
        plt.clf()

def plot_x_y(x_data, y_data, label):
    '''
    This function will plot the given x & y data using line plot
    
    Parameters
    ----------
    x_data : np.array
        X axis data

    y_data : np.array
        Y axis data

    label : str
        label string for plotting (model name)
     
    Returns
    -------
    plt : matplotlib object
        for plotting purposes

    Examples
    --------
    >>> plot_x_y(x_data, y_data, label)
    '''
    plt.plot(x_data, y_data, label=label)
    return plt

def get_result(final_model, test_x, test_y):
    '''
    This function will plot the given x & y data using line plot
    
    Parameters
    ----------
    final_model : sklearn model
        given model from sklearn

    test_x : np.array
        X axis data

    test_y : np.array
        Y axis data
     
    Returns
    -------
    classification report

    Examples
    --------
    >>> get_result(final_model, test_x, test_y)
    '''
    return classification_report(test_y, final_model.predict(test_x), 
        target_names=["Graduate", "Drop"], output_dict=True)


if __name__ == "__main__":
    main(opt["--test"], opt["--out_dir"])