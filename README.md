# Student Dropout Predictor


* Author: Ranjit Sundaramurthi
* Contributors: Andy Wang, Caesar Wong, Ziyi Chen



# Introduction


## Objective


Academic performance/graduation in a population is an important factor in their overall employability which contributes towards economic development. This Data Science project predicts Student Dropout given the factors on demography, socioeconomics, macroeconomics, and relevant academic data provided by the Student on enrollment. This prediction is important to understand the student's academic capacity. This important knowledge can be used to identify key areas of development such as the development of socially disadvantaged communities, improvement of academic programs, development of educational funding programs, etc.  This project will try to investigate the following research question:

> **Given a student with his/her demography, socioeconomics, macroeconomics, and relevant academic data, how accurately can we predict whether he/she will drop out of school?**



## Dataset


The dataset used in the project contains data collected at the time of student enrollment and a snapshot of their performance at the end of the 2nd semester at their respective Universities. This includes discrete and continuous data that capture the various facets of the student. These include macroeconomic factors of inflation, GDP, and the unemployment rate. It covers the personal/family details of the student such as gender, previous grade, educational special needs, financial status, parents' education, and parents' occupation. It captures aspects of the educational system such as coursework enrolled, day/evening classes, scholarships offered, etc. The dataset is created by Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista from the Polytechnic Institue of Portalegre. It was sourced from the UCI Machine Learning Repository and can be downloaded from [here](https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). Each row represents the details pertaining to an individual student and there are no duplicates. 


The original dataset exhibits three classifications (class) of students - Graduate, Enrolled, and Dropout. For the binary classification question pursued in this project, the class Enrolled is omitted from the dataset. The preliminary EDA shows there are 2209 examples of Graduate students and 1421 examples of Dropouts. Thus the dataset imbalance is not a major concern and can be addressed through balancing techniques learned in the MDS program.  


## Analysis Roadmap


We partitioned the dataset into training and test sets (80%: 20%). A detailed EDA is performed to understand the distribution of the 36 features and their correlation. The insights from EDA are used to eliminate features to reduce redundancy. Correlation maps, Bar plots and Pairwise Scatter plots from the EDA are  used on the continuous features such as Inflation rate, GDP, and Unemployment rate to draw inferences for feature selection.    


The modeling is performed using the Naive Bayes, Logistic Regression and Random Forest classification algorithms to identify the best-performing model. The Naive Bayes algorithm is shortlisted for its ability to scale well and handle sparse data. With multiple categorical features, there is sparsity in our model. The Logistic Regression algorithm is chosen for its similar advantages to the Naive Bayes algorithm along with the attractive advantage of providing interpretability for feature importance selection. The Random Forest classifier enabled us to apply ensemble models on the dataset. The performance metrics for our problem statement are `Recall` and `f1 score` respectively, in order of importance. Type 2 errors where actual dropouts are not identified reduce the usefulness of our project. Thus `Recall` is an essential performance metric. The Type 1 errors indicated by the precision of the model are of relatively lesser significance as actual graduates incorrectly classified as dropouts will provide a conservative model which is relatively acceptable.

## Results and Conclusions Roadmap

The hyperparameters of the aforementioned models are optimized using cross-validation to determine the best estimator. The performance of these models is tabulated in the report for comparison. The reasons for the best estimator selection are documented along will the modeling assumptions and identified deficiencies. The train data is refit on the best estimator and the final predictions are made on the test data. The confusion matrix is documented and included in the final report along with comments on misclassifications and their effect on model performance.

The EDA performed can be found in the [dropout_pred_EDA.pdf](https://github.com/UBC-MDS/dropout-predictions/blob/main/src/dropout_pred_EDA.pdf).

## Data Analysis Pipeline

In this project, we adopt the following data analysis pipeline. First of all, we dowload and preprocess the raw data. After splitting and storing the required data files, we use the `train_eda.csv` as the input of `general_EDA.py`, `train.csv` for `model_training.py`, and `testing.py` for `model_result.py`.

![plot](doc/data_analysis_pipeline.png)

## Usage

There are different ways to replicate the analysis.


1. Clone [this](https://github.com/UBC-MDS/dropout-predictions.git) GitHub repository

```
git clone https://github.com/UBC-MDS/dropout-predictions.git
```

2. Navigate to the GitHub repository

```
cd dropout-predictions
```

3. Install the conda environment listed in [here](https://github.com/UBC-MDS/dropout-predictions/blob/main/env/dropout_pred_env.yml) 

```
conda env create -f env/dropout_pred_env.yml
```

4. Activate the environment 

```
conda activate dropout_pred_env
```

We can either use the [Makefile](#makefile) or [Shell Script](#clean-files) to run the analysis.

### Makefile

#### Run All

To run the whole analysis, run the following command in the root directory:

```
make all
```

It will check whether the [final report](doc/The_Report_of_Dropout_Prediction.html) exists or not. If the final report does not exist, the Makefile will run all the dependencies required to generate the report.

#### Clean Files

To clean the intermediate and final results including images, CSV files and report, run the following command in the root directory:

```
make clean
```

It will clean all the files under `data/raw/`, `results/`, and all the CSV files under `data/processed/`.

### Shell Script

After activating the Conda environment, run the following command under the `src` folder.

```
bash data_analysis_pipeline.sh
```

- [Shell Script](src/data_analysis_pipeline.sh) content:

    <<comment
    This shell script will include all the script running required to reproduce the dropout prediction analysis.
    Please run this script within the src/ folder
    comment

    # download data
    python download_data.py --url="https://raw.githubusercontent.com/caesarw0/ml-dataset/main/students_dropout_prediction/data.csv" --extract_to="../data/raw/data.csv"

    # preprocess data 
    python preprocessing.py --input_path="../data/raw/data.csv" --sep=',' --test_size=0.2 --random_state=522 --output_path="../data/processed"

    # generate EDA plot
    python general_EDA.py --input_path="../data/processed/train_eda.csv" --output_path="../results/"

    # model training
    python model_training.py --train="../data/processed/train.csv" --scoring_metrics="recall" --out_dir="../results/"

    # model testing
    python model_result.py --test="../data/processed/test.csv" --out_dir="../results/"

    # report generation
    Rscript -e 'rmarkdown::render("../doc/The_Report_of_Dropout_Prediction.Rmd")'


## License

The Student Dropout Predictor materials here are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for our purpose of academic study and understanding, with the appropriate credit given.


## References

- Realinho,Valentim, Vieira Martins,Mónica, Machado,Jorge & Baptista,Luís. (2021). Predict students' dropout and academic success. UCI Machine Learning Repository. https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

