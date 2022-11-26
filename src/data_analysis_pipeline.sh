# Author: Caesar
# Date: 2022-11-26

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

# generate final report 