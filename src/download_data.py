# Author: Andy Wang
# Date: 2022-11-18

"""
A script that downloads data in website and unzip to the location user specify and get data.csv file.

Usage: src/download_data.py --url=<url> --extract_to=<extract_to> 
 
Options:
--url=<url>             URL from where to download the data (from uci website)
--extract_to=<extract_to>   Specify the path where user can get the data.csv in their local repository
"""

# Example:
# python download_data.py --url="https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip" --extract_to="../data/raw/"

# importing necessary modules
from docopt import docopt
import requests, zipfile
from io import BytesIO
import os
from utils.print_msg import print_msg # adding utils function for print msg


opt = docopt(__doc__) # This would parse into dictionary in python

def main(url, extract_to):
    """
    Download the data from the given url and unzip it to its parent directory 
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
    >>> main("https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip", "../data/raw/")
    """
    
    print_msg("Downloading Started")
    # Split URL to get the file name
    filename = url.split('/')[-1]
    print(filename)

    # Downloading the file by sending the request to the URL
    req = requests.get(url, verify=False)
    print_msg("Downloading Completed")

    # extracting the zip file contents
    zipfile1= zipfile.ZipFile(BytesIO(req.content))
    
    
    try:
        zipfile1.extractall(extract_to)
    
    except:
        os.makedirs(os.path.dirname(extract_to))
        zipfile1.extractall(extract_to)

    print_msg("Download_data Completed - End of script")

if __name__ == "__main__":
    main(opt["--url"], opt["--extract_to"])