# Logistic Regression Comparison: Scratch vs Scikit-Learn  

## Project Overview  
This project implements **Logistic Regression** in two ways:  
1. **From scratch** using only NumPy and pandas
2. Using **Scikit-Learn’s** LogisticRegression class

The main goals of this project are not just to compare two implementations, but also to:  
- **Practice coding** of how logistic regression works internally
- **Practice structuring** a complete data science project from start to finish
- **Gain experience** in exploring, cleaning, and preparing a dataset  

The dataset used is the **Wine Quality dataset (binary: red vs white wine)** from Kaggle, [dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

## Project Structure  
```
├── data/                    
│ ├── data analysis/           # Notebooks for EDA
│ │ └── data_analysis.ipynb
│ ├── processed/               # Processed dataset
│ │ └── clean_wine.csv
│ └── raw/                     # Raw dataset
│   └── winequalityN.csv
│
├── src/                  
│ ├── pycache/               
│ ├── init.py
│ ├── compare_models.py       # Scratch + sklearn models and compare results
│ ├── data_prep.py            # Data preparation (train/test split, shufflw)
│ ├── logisticR_scratch.py    # Logistic Regression from scratch 
│ ├── main_lib.py             # Main script to run sklearn model
│ ├── main_scratch.py         # Main script to run scratch model
│
├── Makefile                  # (Optional) Automate running and instalation
├── report.pdf                # Final report of the project
├── requirements.txt          # Python packages
```

## How to Run
1. **Run Logistic Regression from Scratch**

python src/main_scratch.py or make scratch

2. **Run Logistic Regression with Scikit-Learn**
 
python src/main_lib.py or make library

3. **Compare Both Models**
 
python src/compare_models.py or make compare

This script will output metrics like Accuracy, Log Loss, ROC AUC, Confusion Matrix, and generate a ROC Curve plot

## Report
The full analysis, results, and conclusions are documented in:

report.pdf

