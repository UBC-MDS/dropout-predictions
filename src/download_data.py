# importing necessary modules
import requests, zipfile
from io import BytesIO
print('Downloading started')


def download_and_unzip(url, extract_to='../data/raw/'):
    # Split URL to get the file name
    filename = url.split('/')[-1]
    print(filename)

    # Downloading the file by sending the request to the URL
    req = requests.get(url)
    print('Downloading Completed')

    # extracting the zip file contents
    zipfile= zipfile.ZipFile(BytesIO(req.content))
    zipfile.extractall(extract_to)

download_and_unzip(url='https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip')