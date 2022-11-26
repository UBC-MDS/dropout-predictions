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
# python general_EDA.py --input_path="../data/processed/train_eda.csv" --output_path="../results/"

# importing necessary modules
from docopt import docopt

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from hashlib import sha1
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.model_selection import train_test_split

from utils.save_plot import save_chart # adding utils function for plot saving
from utils.print_msg import print_msg # adding utils function for print msg

opt = docopt(__doc__) # This would parse into dictionary in python


def main(input_path, output_path):
    print_msg("Begin General EDA")
    
    #creating target count bar plot
    df = pd.read_csv(input_path)
    print("df shape: ")
    print(df.shape)

    assert df.shape[1] == 37, "Incorrect number of column expected"

   
    # 1. Creating first plot
    print_msg("Constructing Plot 1")
    target_bar = alt.Chart(df,
                     title='Target Count Bar Plot'
                ).mark_bar().encode(
                        x='count(Target)',
                        y=alt.Y('Target', sort='-x')
                )
    # Saving the first plot
    save_chart(target_bar, output_path + 'target_count_bar_plot.png',2)

    
    # 2. creating feature correlation plot
    print_msg("Constructing Plot 2")
    df_target = df.replace({'Target': {'Dropout': 1, 'Graduate': 0}})
    f = plt.figure(figsize=(15, 15))
    sns.heatmap(df_target.corr(),annot=False, cmap='coolwarm',center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .7}).set(
    title="Correlation Heatmap")
    # Saving the feature correlation plot
    plt.savefig(output_path + "correlation_heatmap_plot.png", bbox_inches = 'tight')

    # 3. creating correlation heatmap
    print_msg("Constructing Plot 3")
    feat_corr = df_target.drop("Target", axis=1).apply(lambda x: x.corr(df_target.Target))
    feat_corr = pd.DataFrame(feat_corr, columns=['correlation']).sort_values(['correlation'], ascending=False)
    
    # saving correlation heatmap
    plt.figure(figsize=(10,8))
    sns.barplot(x=feat_corr['correlation'], y=feat_corr.index, palette="vlag").set(
    title="Feature Correlation for Target", xlabel="Feature Correlation",
    ylabel="Feature Names")
    plt.savefig(output_path + "correlation_with_target_plot.png", bbox_inches = 'tight')
    
    
    # 4. creating age at enrollment density plot by gender & dropout
    print_msg("Constructing Plot 4")
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
    # saving gender density plot with age at enrollment
    save_chart(gender_density, output_path + 'gender_density_plot.png',2)
    print_msg("General EDA Completed - End of general_EDA")

if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])