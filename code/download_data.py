# Author: Andy Wang
# Date: 2022-11-18

"""
A script that downloads data in website and unzip to the location user specify and get data.csv file.
​
Usage: src/download_data.py --url=<url> --extract_to=<extract_to> 
 
Options:
--url=<url>             URL from where to download the data (from uci website)
--extract_to=<extract_to>   Specify the path where user can get the data.csv in their local repository
"""

# Example:
# python code/download_data.py --url="https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip" --extract_to="./data/raw/"


# importing necessary modules
from docopt import docopt
import requests, zipfile
from io import BytesIO

opt = docopt(__doc__) # This would parse into dictionary in python
"""
    Download the data from the given url and unzip it to its parent directory.
​
    Parameters:
    url (str): The raw zip url which includes data.csv
    extract_to (str):  Path of where to get the file with downloaded data locally
​
    Returns:
    Stores the data.csv file in the extract_to's parent directory
    Example:
    main("https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip", "../data/raw/")
    """


def main(url, extract_to):
    print('Downloading started')
    # Split URL to get the file name
    filename = url.split('/')[-1]
    print(filename)

    # Downloading the file by sending the request to the URL
    req = requests.get(url, verify=False)
    print('Downloading Completed')

    # extracting the zip file contents
    zipfile1= zipfile.ZipFile(BytesIO(req.content))
    zipfile1.extractall(extract_to)

if __name__ == "__main__":
    main(opt["--url"], opt["--extract_to"])