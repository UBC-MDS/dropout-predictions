# Author: Ziyi Chen
# Date: 2022-11-24

"""
A script that uses preprocessed data and generate corresponding plots under the result/eda/ folder.
Usage: src/general_EDA.py --input_path=<input_path> --output_path=<output_path> 
 
Options:
--input_path=<input_path>       Input path for the preprocessed dataset

--output_path=<output_path>     Specify the path where user can store the plots
"""

# Example:
# python general_EDA.py --input_path="../data/processed" --output_path="../result/eda/"

# importing necessary modules
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from hashlib import sha1
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.model_selection import train_test_split

opt = docopt(__doc__) # This would parse into dictionary in python

import vl_convert as vlc 


def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")


def main(input_path, output_path):
    
    #creating target count bar plot
    df = pd.read_csv(input_path)
    target_bar = alt.Chart(df,
                     title='Target Count Bar Plot'
         ).mark_bar().encode(
                x='count(Target)',
                y=alt.Y('Target', sort='-x'))
    save_chart(target_bar, 'plot1.png',2)

    #creating feature correlation plot
    df_target = df.replace({'Target': {'Dropout': 1, 'Graduate': 0}})
    f = plt.figure(figsize=(15, 15))
    sns.heatmap(df_target.corr(),annot=False, cmap='coolwarm',center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .7})
    plt.savefig("plot2.png")

    #creating correlation heatmap
    feat_corr = df_target.drop("Target", axis=1).apply(lambda x: x.corr(df_target.Target))
    feat_corr = pd.DataFrame(feat_corr, columns=['correlation']).sort_values(['correlation'], ascending=False)

    plt.figure(figsize=(10,8))
    sns.barplot(x=feat_corr['correlation'], y=feat_corr.index, palette="vlag")
    plt.savefig("plot3.png")
    
    #creating age at enrollment density plot by gender & dropout
    gender_dict = {1: 'male', 0: 'female'}
    df2=df.replace({"Gender": gender_dict})
    gender_density = (alt.Chart(df2)
    .transform_density(
     'Age at enrollment',
     groupby=['Target','Gender'],
     as_=['Age at enrollment', 'density'],
     counts=True,
    )
    .mark_line().encode(
     x='Age at enrollment',
     y='density:Q',
     color='Target',
    tooltip='Age at enrollment')
    .facet('Gender',
       title="Age at Enrollment Density Plot by Gender & Dropout"
       )
    )
    save_chart(gender_density, 'plot4.png',2)

if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])