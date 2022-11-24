# Author: Andy Wang
# Date: 2022-11-18

"""
A script that downloads data in website and unzip to the location user specify and get data.csv file.

Usage: src/general_EDA_draft.py --input_path=<input_path> --output_path=<output_path>
 
Options:
--input_path=<input_path>       Input data path (e.g. data/raw/data.csv)
--output_path=<output_path>     Output plot / table path (e.g. src/eda/)
"""

# Example:
# python download_data.py --url="https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip" --extract_to="./data/raw/"

# importing necessary modules
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.model_selection import train_test_split


opt = docopt(__doc__) # This would parse into dictionary in python

def plot_1():
    return

def plot_2():
    return

def main(input_path, output_path):
    return

if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])