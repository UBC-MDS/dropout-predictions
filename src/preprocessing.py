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

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


opt = docopt(__doc__) # This would parse into dictionary in python

def main(input_path, sep, test_size, random_state, output_path):
    
    df = pd.read_csv(input_path, sep=sep)
    # validate df
        # check shape, etc.
    
    assert 'Nacionality' in df.columns, "typo is missing"

    # rename column & fix typo
    df = df.rename(columns={'Nacionality': 'Nationality', 'Daytime/evening attendance\t': 'Daytime_evening_attendance'})

    assert 'Nationality' in df.columns, "failed to fix the typo"

    # drop na from df
    df = df.dropna()
    
    # drop enrolled student
    df = df.drop(df[df.Target == 'Enrolled'].index)

    # rearrange column
    df = df[['Marital status', 'Nationality','Displaced','Gender',
            'Age at enrollment', 'International',
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation",
                
            'Educational special needs', 'Debtor',
            'Tuition fees up to date',  'Scholarship holder',
            'Unemployment rate','Inflation rate', 'GDP',
                        
            'Application mode', 'Application order', 'Course',
                        
            'Daytime_evening_attendance', 'Previous qualification',
            'Previous qualification (grade)', 
            'Admission grade',
                
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)', 'Target']]

    print("df shape : ")
    print(df.shape)

    # data splitting
    try:
        train_df, test_df = train_test_split(df, test_size=float(test_size), random_state=int(random_state))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    
    print("train shape : ")
    print(train_df.shape)

    # storing EDA ready dataset (1. dropped 'enrolled', 2. Fixed column typo, 3. dropna)
    train_df.to_csv(output_path + '/train_eda.csv', index=False)

    # perform column transformation
    # - binary
    # - categorical (OHE)
    # - continuous (standardscaler)
    numeric_features = ['Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)',
                        'Age at enrollment',
                        'Curricular units 1st sem (approved)',
                        'Curricular units 2nd sem (approved)'
                        ]
    binary_features = ['Gender', 'Debtor', 'Scholarship holder', 'Tuition fees up to date']
    categorical_features = ['Application mode']

    drop_features = ['Marital status', 'Nationality','Displaced',
            'International',
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation",

            'Educational special needs',
            'Unemployment rate','Inflation rate', 'GDP',

            'Application order', 'Course',

            'Daytime_evening_attendance', 'Previous qualification',
            'Previous qualification (grade)', 
            'Admission grade',

            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (without evaluations)']

    numeric_transformer = StandardScaler()
    binary_transformer = OneHotEncoder(drop="if_binary", dtype=int)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)


    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (binary_transformer, binary_features),    
        (categorical_transformer, categorical_features),
        ("drop", drop_features),
            ("passthrough", ['Target'])
    )

    # fit preprocessor & transform training data
    train_data = preprocessor.fit_transform(train_df)
    ohe_feats = preprocessor.named_transformers_['onehotencoder-1'].get_feature_names(binary_features).tolist()
    categ_feats = preprocessor.named_transformers_['onehotencoder-2'].get_feature_names(categorical_features).tolist()

    feature_names = numeric_features + ohe_feats + categ_feats + ['Target']
    transformed_train_df = pd.DataFrame(train_data, columns=feature_names)

    # transform testing data
    test_data = preprocessor.transform(test_df)
    transformed_test_df = pd.DataFrame(test_data, columns=feature_names)

    # convert target to 1 (dropout) or 0 (graduate)
    target_dict = {'Dropout': 1, 'Graduate': 0}
    transformed_train_df = transformed_train_df.replace({'Target': target_dict})
    transformed_test_df = transformed_test_df.replace({'Target': target_dict})

    transformed_train_df.to_csv(output_path + '/train.csv', index=False)
    transformed_test_df.to_csv(output_path + '/test.csv', index=False)

    print("Finished preprocessing")

    

if __name__ == "__main__":
    main(opt["--input_path"], opt["--sep"], opt["--test_size"], opt["--random_state"], opt["--output_path"])