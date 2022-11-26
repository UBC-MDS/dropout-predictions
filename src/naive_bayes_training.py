# Author:Ranjit Sundaramurthi
# Date

"""
A script that train and find the best model and store it under the results/ folder.

Usage: src/naive_bayes_training.py --train=<train> --out_dir=<out_dir>
 
Options:
--train=<train>       Input path for the train dataset
--out_dir=<out_dir>   Path to directory where the serialized model should be written

"""

# Example:
# python training.py --train="../data/processed/train.csv" --out_dir="../results/"

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
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

# logistic regression and random forest
opt = docopt(__doc__)  # This would parse into dictionary in python


def main(train, out_dir):
    cross_val_results = {}
    # get training data
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=["Target"]), train_df["Target"]

    NB = GaussianNB()
    
    # output the model to local
    # results_dict["NB_f1"] = mean_std_cross_val_scores(
    #     make_pipeline(NB),
    #     X_train,
    #     y_train,
    #     cv = 10,
    #     scoring = 'f1',
    #     return_train_score = True,
    #     n_jobs = -1,
    #     verbose = 1
    # )
    
    # fit the model
    NB.fit(X_train, y_train)
    final_model_log = NB
    
    cross_val_results = output_object(NB, cross_val_results, X_train, y_train)
    
    # save file
    try:
        file_log = open(out_dir + "/model_try_Naive_Bayes", "wb")
    except:
        os.makedirs(os.path.dirname(out_dir))
        file_log = open(out_dir + "/model_try_Naive_Bayes", "wb")


    file_model_result = open(out_dir + "/model_try_Naive_Bayes", "wb")
    pickle.dump(final_model_log, file_model_result)


def output_object(best_model, result_dict, train_x, train_y):
    scoring_metrics = "f1"
    result_dict["NB_f1"] = (
        pd.DataFrame(
            cross_validate(
                best_model,
                train_x,
                train_y,
                return_train_score=True,
                scoring=scoring_metrics,
            )
        )
        .agg(["mean", "std"])
        .round(3)
        .T
    )

    return result_dict


if __name__ == "__main__":
    main(opt["--train"], opt["--out_dir"])
