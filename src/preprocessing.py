# Author: Caesar Wong
# Date: 2022-11-23

"""
A script that preprocess the data and store it under the data/preprocess/ folder.

Usage: src/preprocessing.py --input_path=<input_path> --sep=<sep> --test_size=<test_size> --random_state=<random_state> --output_path=<output_path> 
 
Options:
--input_path=<input_path>       Input path for the raw dataset
--sep=<sep>                     Separator for reading the CSV file
--test_size=<test_size>         Test size for train_test_split
--random_state=<random_state>   Random state for train_test_split
--output_path=<output_path>     Specify the path where user can store the preprocessed dataframe

"""

# Example:
# python preprocessing.py --input_path="../data/raw/data.csv" --sep=';' --test_size=0.2 --random_state=522 --output_path="../data/processed"

# importing necessary modules
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os

from sklearn.model_selection import train_test_split
import pandas as pd

opt = docopt(__doc__) # This would parse into dictionary in python

def main(input_path, sep, test_size, random_state, output_path):
    
    df = pd.read_csv(input_path, sep=sep)
    # validate df
        # check shape, etc.

    # rename column & fix typo
    df = df.rename(columns={'Nacionality': 'Nationality', 'Daytime/evening attendance\t': 'Daytime_evening_attendance'})

    # drop na from df
    df = df.dropna()
    
    # drop enrolled student
    df = df.drop(df[df.Target == 'Enrolled'].index)

    print("df shape : ")
    print(df.shape)

    train_df, test_df = train_test_split(df, test_size=float(test_size), random_state=int(random_state))

    print("train shape : ")
    print(train_df.shape)

    train_df.to_csv(output_path + '/train.csv', index=False)
    test_df.to_csv(output_path + '/test.csv', index=False)

    print("Finished data splitting to csv")

    

if __name__ == "__main__":
    main(opt["--input_path"], opt["--sep"], opt["--test_size"], opt["--random_state"], opt["--output_path"])