# Kaggle Competition Submission - Recipe Rating Prediction

## Overview
This repository contains the Jupyter Notebook used for my submission to a Kaggle competition as part of the coursework for my B.Sc. in Data Science and Programming at IIT Madras. The competition involved predicting food recipe ratings based on various features such as user interactions, recipe reviews, timestamps, and sentiment analysis.

## Objective
The goal was to predict the rating (0-5) of recipes using machine learning models trained on the provided dataset. The project focused on handling imbalanced data, extracting meaningful features, and optimizing model performance.

## Key Features of the Notebook

### 1. Data Loading and Exploration
- Loaded training and testing datasets.
- Conducted exploratory data analysis (EDA) to understand feature distributions, correlations, and missing values.
- Visualized data using histograms, boxplots, and heatmaps.

### 2. Data Preprocessing
- Handled missing values by imputing or dropping them.
- Extracted features from timestamps (e.g., `Hour`, `TimeOfDay_Influence`).
- Engineered new features such as sentiment polarity, positive interaction ratios, and social influence metrics.
- Preprocessed text data (`Recipe_Review`) by cleaning HTML tags, URLs, and non-alphabetic characters.

### 3. Feature Engineering
- Added features like:
  - `Emoticons_Count`, `Exclamation_Count`, `Capital_Letters_Count`
  - `NormalizedRecipeNumber`, `Weighted_SocialInfluence`
  - Sentiment-based metrics (`Polarity_Sentiment`)
- Used techniques like one-hot encoding and vectorization (`CountVectorizer`) for text data.

### 4. Model Development
Implemented multiple models to predict ratings:
- **Random Forest Classifier**  
  - Achieved an accuracy of **77.52%** after hyperparameter tuning.
- **K-Nearest Neighbors (KNN)**  
  - Achieved an accuracy of **76.11%** after hyperparameter tuning.
- **XGBoost Classifier**  
  - Achieved an accuracy of **77.49%** after hyperparameter tuning.
- **LightGBM Classifier (LGBM) (Final Model)**  
  - Achieved the best accuracy (**78%**) after extensive hyperparameter tuning.
  - Outperformed other models due to its ability to handle imbalanced datasets effectively.

### 5. Hyperparameter Tuning
- Used `RandomizedSearchCV` to optimize model parameters across all models.
- Explored parameters such as learning rate, max depth, number of estimators, subsample fraction, etc.

### 6. Model Comparison
- Compared performance across models:
  - **LightGBM > XGBoost > Random Forest > KNN**
- Highlighted why ensemble methods performed better than KNN on this dataset.

### 7. Submission
- Generated predictions for the test dataset using the tuned LightGBM model.
- Saved predictions in `submission.csv` for Kaggle leaderboard evaluation.

## Results
The final submission achieved a competitive score on the Kaggle leaderboard, showcasing the effectiveness of feature engineering and model optimization techniques applied in this notebook.

## Requirements
### Libraries Used:
#### Python Libraries:
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `re`
#### Machine Learning Libraries:
- `scikit-learn` (`RandomForestClassifier`, `KNeighborsClassifier`)
- `XGBoost`
- `LightGBM`
#### Text Processing:
- `CountVectorizer`
#### Imbalanced Data Handling:
- `SMOTE` (`imbalanced-learn`)

Install all dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
```

## File Structure
- **Notebook File**: Contains all code for data preprocessing, model development, tuning, and predictions.
- **Dataset Files**: Training and testing datasets provided by Kaggle.
- **Submission File**: Final predictions saved as `submission.csv`.

## How to Run
1. Clone this repository or download the notebook file.
2. Ensure all required libraries are installed.
3. Place the dataset files (`train.csv` and `test.csv`) in the working directory.
4. Run the notebook step-by-step to reproduce results or modify it for further experimentation.

## Position on Kaggle Leaderboard
I finished **32nd among 950 participants**, ranking in the **top 3%**.

## Acknowledgments
This project was completed as part of my coursework at IIT Madras under the B.Sc. program in Data Science and Programming. Special thanks to my mentors and peers who guided me throughout this journey!

Feel free to reach out if you have any questions or suggestions! 😊
