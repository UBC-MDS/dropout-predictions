# Author: Andy Wang, Ranjit Sundaramurthi, Caesar
# Date: 2022-11-25

"""
A script that train all the models, and find the best model (for those model having hyperparameter tuning).
Finally, storing all the model, cv result under the results/ folder.

Usage: src/model_training.py --train=<train> --scoring_metrics=<scoring_metrics> --out_dir=<out_dir>
 
Options:
--train=<train>                     Input path for the train dataset
--scoring_metrics=<scoring_metrics> Scoring Metrics for classification (e.g. 'f1', 'recall', 'precision')
--out_dir=<out_dir>                 Path to directory where the serialized model should be written

"""

# Example:
# python model_training.py --train="../data/processed/train.csv" --scoring_metrics="f1" --out_dir="../results/"

# import
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os
import pickle
import pandas as pd
from scipy.stats import lognorm, loguniform, randint
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from utils.print_msg import print_msg # adding utils function for print msg


# logistic regression and random forest
opt = docopt(__doc__) # This would parse into dictionary in python

def main(train, scoring_metrics, out_dir):
    '''
    Run hyperparameter optimization on logistic regression and random forest,
    Naive Bayes dont have search CV
    Perform model training & saving
    
    Parameters
    ----------
    train : str
        training data path

    scoring_metrics : str
        scoring metrics name, e.g. 'f1', 'recall'

    out_dir : str
        output directory path
     
    Returns
    -------
    store 3 model to results/ folder
        

    Examples
    --------
    >>> main(opt["--train"], opt["--scoring_metrics"], opt["--out_dir"])
    '''
    print_msg("Begin Model Training")
    cross_val_results = {}
    # get training data
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=["Target"]), train_df["Target"]


    # initialize our selected model
    models = {'logisticRegression': LogisticRegression(random_state=123, max_iter=1000),
              'NaiveBayes': GaussianNB(),
              'RandomForestClassifier': RandomForestClassifier(random_state=123)}
    
    # random forest specific hyperparameter
    max_features = randint(10, 26) # number of features in consideration at every split
    max_depth = randint(10, 50) # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node

    params = {'logisticRegression': {"class_weight": [None, 'balanced'],
                                        "C": loguniform(1e-3, 1e3)
                                    },
            'RandomForestClassifier':   {
                                        'max_features': max_features,
                                        'max_depth': max_depth,
                                        'min_samples_split': min_samples_split,
                                        'min_samples_leaf': min_samples_leaf
                                    }         
            }
    
    # for each model 
    for model_name in models:
        print_msg(model_name + " Training")
        if model_name != 'NaiveBayes':
            cur_pipe = hyper_tuning(models[model_name], scoring_metrics, params[model_name])
        else:
            cur_pipe = models[model_name]

        cur_pipe.fit(X_train, y_train)
        cross_val_results[model_name] = output_object(cur_pipe, X_train, y_train, scoring_metrics)

        # save file
        try:
            file_log = open(out_dir + '/model_' + model_name, 'wb')
        except:
            os.makedirs(os.path.dirname(out_dir))
            file_log = open(out_dir + '/model_' + model_name, 'wb')
        pickle.dump(cur_pipe, file_log)

        # obtain best parameter set except NaiveBayes
        if model_name != 'NaiveBayes':
            print(cur_pipe.best_params_)
        print_msg(model_name + " Training Completed")

        
        
    # Storing CV result
    cv_result = pd.concat(cross_val_results, axis=1)
    print(cv_result)
    cv_result.to_csv(out_dir + '/cv_result.csv')

    print_msg("Model Training Completed - End of Training")

    

# create hyper tuning search sklearn object
def hyper_tuning(model, scoring_metrics, param_dist):
    '''
    return the randomized search object
    
    Parameters
    ----------
    model : sklearn.model
        dummy model

    scoring_metrics : str
        scoring metrics name, e.g. 'f1', 'recall'

    param_dist : dict
        hyperparameter distribution for randomized search
     
    Returns
    -------
    random_search : sklearn randomized search cv
        

    Examples
    --------
    >>> cur_pipe = hyper_tuning(models[model_name], scoring_metrics, params[model_name])
    '''
    # output the best model to local
    random_search = RandomizedSearchCV(
                                        model,
                                        param_dist,
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=1,
                                        random_state=123,
                                        return_train_score=True,
                                        scoring=scoring_metrics
                                    )
    
    return random_search
     
def output_object(best_model, train_x, train_y, scoring_metrics):
    '''
    given the best model, scoring metrics and training data, 
    return the aggregated (mean, std) cross_validate result
    
    Parameters
    ----------
    best_model : sklearn.model
        best performing model / dummy model

    train_x : pd.DataFrame
        feature of the training data

    train_y : pd.DataFrame
        target of the training data

    scoring_metrics : str
        scoring metrics name, e.g. 'f1', 'recall'
    
    Returns
    -------
    pd.DataFrame
        a dataframe of the aggregated cross_validate result

    Examples
    --------
    >>> cross_val_results[model_name] = output_object(cur_pipe, X_train, y_train, scoring_metrics)
    '''
    return pd.DataFrame(cross_validate(
                        best_model, train_x, train_y, return_train_score=True, 
                        scoring=scoring_metrics)
                        ).agg(['mean', 'std']).round(3).T
    
    
if __name__ == "__main__":
    main(opt["--train"], opt["--scoring_metrics"], opt["--out_dir"])