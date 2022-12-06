# Author: Andy Wang
FROM jupyter/scipy-notebook

USER root

RUN apt-get update

# Install Python 3 packages
RUN conda install -c conda-forge --quiet --yes \
    'docopt==0.6.*' 

# Install R packages
RUN conda install -c conda-forge --quiet --yes \
    'r-base=4.1.2' \
    'r-rmarkdown' \
    'r-tidyverse=1.3*' \
    'r-knitr' \
    'r-kableextra'

RUN pip install vl-convert-python
