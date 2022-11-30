# Makefile
# Author: Caesar Wong
# Date: 2022-11-29

# ** please activate the conda environment before using the Make

all: doc/The_Report_of_Dropout_Prediction.html 
#data/raw/data.csv data/processed/train_eda.csv results/target_count_bar_plot.png results/correlation_heatmap_plot.pn results/correlation_with_target_plot.png results/gender_density_plot.png

# download data
data/raw/data.csv: src/download_data.py
	python src/download_data.py --url="https://raw.githubusercontent.com/caesarw0/ml-dataset/main/students_dropout_prediction/data.csv" --extract_to="data/raw/data.csv"

# preprocess data
data/processed/train_eda.csv data/processed/train.csv data/processed/test.csv: data/raw/data.csv src/preprocessing.py
	python src/preprocessing.py --input_path="data/raw/data.csv" --sep=',' --test_size=0.2 --random_state=522 --output_path="data/processed"

# generate EDA plot
results/target_count_bar_plot.png results/correlation_heatmap_plot.pn results/correlation_with_target_plot.png results/gender_density_plot.png: data/processed/train_eda.csv src/general_EDA.py
	python src/general_EDA.py --input_path="data/processed/train_eda.csv" --output_path="results/"

# model training
results/cv_result.csv results/model_NaiveBayes results/model_RandomForestClassifier results/model_logisticRegression: data/processed/train.csv src/model_training.py 
	python src/model_training.py --train="data/processed/train.csv" --scoring_metrics="recall" --out_dir="results/"

# model testing
results/Confusion_Matrix_logisticRegression.png results/Confusion_Matrix_NaiveBayes.png results/Confusion_Matrix_RandomForestClassifier.png results/PR_curve.png results/ROC_curve.png results/score_on_test.csv: src/model_result.py data/processed/test.csv results/cv_result.csv results/model_NaiveBayes results/model_RandomForestClassifier results/model_logisticRegression
	python src/model_result.py --test="data/processed/test.csv" --out_dir="results/"

# generate html report
doc/The_Report_of_Dropout_Prediction.html : doc/The_Report_of_Dropout_Prediction.Rmd doc/dropout_prediction_references.bib results/Confusion_Matrix_logisticRegression.png results/Confusion_Matrix_NaiveBayes.png results/Confusion_Matrix_RandomForestClassifier.png results/PR_curve.png results/ROC_curve.png results/score_on_test.csv results/target_count_bar_plot.png results/correlation_heatmap_plot.pn results/correlation_with_target_plot.png results/gender_density_plot.png
	Rscript -e 'rmarkdown::render("doc/The_Report_of_Dropout_Prediction.Rmd")'

clean:
	rm -rf data/raw/*
	rm -f data/processed/*.csv
	rm -f results/*