# Author: Caesar
# Date: 2022-11-26

<<comment
This shell script will include all the script running required to reproduce the dropout prediction analysis.
Please run this script within the src/ folder
comment

# download data
python download_data.py --url="https://archive-beta.ics.uci.edu/static/ml/datasets/697/predict+students+dropout+and+academic+success.zip" --extract_to="../data/raw/"
    
# preprocess data 
python preprocessing.py --input_path="../data/raw/data.csv" --sep=';' --test_size=0.2 --random_state=522 --output_path="../data/processed"

# generate EDA plot
python general_EDA.py --input_path="../data/processed/train_eda.csv" --output_path="../results/"

# model training
python training.py --train="../data/processed/train.csv" --scoring_metrics="f1" --out_dir="../results/"

# model testing
python testing.py --test="../data/processed/test.csv" --out_dir="../results/"

# generate final report 