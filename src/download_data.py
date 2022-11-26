# Author: Andy Wang
# Date: 2022-11-18

"""
A script that downloads data in website and unzip to the location user specify and get data.csv file.

Usage: src/download_data.py --url=<url> --extract_to=<extract_to> 
 
Options:
--url=<url>             URL from where to download the csv file
--extract_to=<extract_to>   Specify the path where user can get the data.csv in their local repository
"""

# Example:
# python download_data.py --url="https://raw.githubusercontent.com/caesarw0/ml-dataset/main/students_dropout_prediction/data.csv" --extract_to="../data/raw/data.csv"

# importing necessary modules
import os
import pandas as pd
from docopt import docopt
import requests
from utils.print_msg import print_msg # adding utils function for print msg


opt = docopt(__doc__) # This would parse into dictionary in python

def main(url, extract_to):
    """
    Download the data from the given url with csv and save it to its parent directory 
    if dirctory is not exist it will create a new directory based on extract_to argument.
    
    Parameters
    ----------
    url (str): The raw zip url which includes data.csv
    extract_to (str):  Path of where to get the file with downloaded data locally
    
    Returns
    -------
    Stores the data.csv file in the extract_to's parent directory
        
    Examples
    --------
    >>> main("https://raw.githubusercontent.com/caesarw0/ml-dataset/main/students_dropout_prediction/data.csv", "../data/raw/data.csv")
    """

    # A try catch block to check url is valid or not 
    try: 
        print_msg("Downloading Started")
        request = requests.get(url)
        request.status_code == 200
        print_msg("Downloading Completed")
    except Exception as req:
        print_msg(req)
        print_msg("Website at the provided url does not exist")


    data = pd.read_csv(url, header=None) # reading the data in a pandas dataframe

    # Save the data as a csv file to the targetted path.
    try:
        print_msg("Loading CSV Started")
        data.to_csv(extract_to, index=False)
        print_msg("Loading CSV Completed")
    except:
        os.makedirs(os.path.dirname(extract_to))
        data.to_csv(extract_to, index=False)
        print_msg("Created new path and loading CSV Completed")


if __name__ == "__main__":
    main(opt["--url"], opt["--extract_to"])