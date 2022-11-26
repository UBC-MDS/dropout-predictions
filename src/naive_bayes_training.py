# Author:Ranjit Sundaramurthi
# Date

"""
A script that train and find the best model and store it under the results/ folder.

Usage: src/training.py --train=<train> --out_dir=<out_dir>
 
Options:
--train=<train>       Input path for the train dataset
--out_dir=<out_dir>   Path to directory where the serialized model should be written

"""

# Example:
# python training.py --train="../data/processed/train.csv" --out_dir="../result/"

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

# logistic regression and random forest
opt = docopt(__doc__)  # This would parse into dictionary in python


def main(train, out_dir):
    cross_val_results = {}
    # get training data
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=["Target"]), train_df["Target"]

    log_reg = LogisticRegression(random_state=123, max_iter=1000)
    # hyper-parameter tuning for two models
    param_dist_log = {"class_weight": [None, "balanced"], "C": loguniform(1e-3, 1e3)}

    # output the best model to local
    random_search_log = hyper_tuning(log_reg, param_dist_log)
    random_search_log.fit(X_train, y_train)
    final_model_log = random_search_log
    cross_val_results = output_object(
        final_model_log, cross_val_results, X_train, y_train
    )

    # save file
    try:
        file_log = open(out_dir + "/model_try", "wb")
    except:
        os.makedirs(os.path.dirname(out_dir))
        file_log = open(out_dir + "/model_try", "wb")

    pickle.dump(final_model_log, file_log)

    file_model_result = open(out_dir + "/model_result_try", "wb")
    pickle.dump(cross_val_results, file_model_result)


def hyper_tuning(model, param_dist):
    scoring_metrics = "f1"

    # output the best model to local
    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=10,
        n_jobs=-1,
        verbose=1,
        random_state=123,
        return_train_score=True,
        scoring=scoring_metrics,
    )

    return random_search


def output_object(best_model, result_dict, train_x, train_y):
    scoring_metrics = "f1"
    result_dict["logreg"] = (
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
