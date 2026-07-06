#Applied AI & ML Essentials Capstone
#Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate
#Student Name: Aashika R
#Dataset: Medical Cost Personal Dataset
#Download the medical cost personal dataset from Kaggle
--------------------------------------------------------------------------------
# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("cleaned_data.csv has been created successfully!")

Output:
cleaned_data.csv has been created successfully!
--------------------------------------------------------------------------------
import os

print(os.listdir())

Output:
['.config', 'scatter_plot.png', 'box_plot.png', 'archive.zip', 'cleaned_data.csv', 'correlation_heatmap.png', 'extracted_data', 'histogram.png', 'line_plot.png', 'bar_chart.png', 'sample_data']
--------------------------------------------------------------------------------

from google.colab import files

files.download("cleaned_data.csv")
--------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv("cleaned_data.csv")
--------------------------------------------------------------------------------
# TASK 1
# Load the cleaned dataset and define X, y_reg and y_clf

# Import pandas
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Display first five rows
print("="*60)
print("FIRST FIVE ROWS")
print("="*60)
print(df.head())

# Display data types
print("\n")
print("="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

# Display dataset shape
print("\n")
print("="*60)
print("DATASET SHAPE")
print("="*60)
print(df.shape)

# Feature Matrix (X)
X = df.drop("charges", axis=1)

# Regression Label
y_reg = df["charges"]

# Classification Label
# Charges greater than median = 1
# Charges less than or equal to median = 0
y_clf = (y_reg > y_reg.median()).astype(int)

# Display information
print("\nRegression Target:")
print(y_reg.head())

print("\nClassification Target:")
print(y_clf.head())

print("\nFeature Matrix:")
print(X.head())

Output:
============================================================
FIRST FIVE ROWS
============================================================
   age     sex     bmi  children smoker     region      charges
0   19  female  27.900         0    yes  southwest  16884.92400
1   18    male  33.770         1     no  southeast   1725.55230
2   28    male  33.000         3     no  southeast   4449.46200
3   33    male  22.705         0     no  northwest  21984.47061
4   32    male  28.880         0     no  northwest   3866.85520


============================================================
DATA TYPES
============================================================
age           int64
sex          object
bmi         float64
children      int64
smoker       object
region       object
charges     float64
dtype: object


============================================================
DATASET SHAPE
============================================================
(1337, 7)

Regression Target:
0    16884.92400
1     1725.55230
2     4449.46200
3    21984.47061
4     3866.85520
Name: charges, dtype: float64

Classification Target:
0    1
1    0
2    0
3    1
4    0
Name: charges, dtype: int64

Feature Matrix:
   age     sex     bmi  children smoker     region
0   19  female  27.900         0    yes  southwest
1   18    male  33.770         1     no  southeast
2   28    male  33.000         3     no  southeast
3   33    male  22.705         0     no  northwest
4   32    male  28.880         0     no  northwest
-------------------------------------------------------------------------------
# TASK 2
# Encode Categorical Columns
# Check categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

print("="*60)
print("CATEGORICAL COLUMNS")
print("="*60)
print(categorical_columns)

# One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\n")

print("="*60)
print("ENCODED FEATURE MATRIX")
print("="*60)
print(X.head())

print("\n")

print("="*60)
print("NEW SHAPE OF DATASET")
print("="*60)
print(X.shape)

print("\n")

print("="*60)
print("ENCODED COLUMN NAMES")
print("="*60)
print(X.columns)

Output:

============================================================
CATEGORICAL COLUMNS
============================================================
Index(['sex', 'smoker', 'region'], dtype='object')


============================================================
ENCODED FEATURE MATRIX
============================================================
   age     bmi  children  sex_male  smoker_yes  region_northwest  \
0   19  27.900         0     False        True             False   
1   18  33.770         1      True       False             False   
2   28  33.000         3      True       False             False   
3   33  22.705         0      True       False              True   
4   32  28.880         0      True       False              True   

   region_southeast  region_southwest  
0             False              True  
1              True             False  
2              True             False  
3             False             False  
4             False             False  


============================================================
NEW SHAPE OF DATASET
============================================================
(1337, 8)


============================================================
ENCODED COLUMN NAMES
============================================================
Index(['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest',
       'region_southeast', 'region_southwest'],
      dtype='object')
--------------------------------------------------------------------------------
# TASK 3
# Train-Test Split and Feature Scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regression Dataset Split

X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Classification Dataset Split

X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

print("="*60)
print("TRAIN TEST SPLIT")
print("="*60)

print("Regression Training Shape :", X_train_reg.shape)
print("Regression Testing Shape  :", X_test_reg.shape)

print("\nClassification Training Shape :", X_train_clf.shape)
print("Classification Testing Shape  :", X_test_clf.shape)

# Standard Scaling for Regression

scaler_reg = StandardScaler()

X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Standard Scaling for Classification

scaler_clf = StandardScaler()

X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)

X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("\nFeature Scaling Completed Successfully.")

Output:
============================================================
TRAIN TEST SPLIT
============================================================
Regression Training Shape : (1069, 8)
Regression Testing Shape  : (268, 8)

Classification Training Shape : (1069, 8)
Classification Testing Shape  : (268, 8)

Feature Scaling Completed Successfully.
--------------------------------------------------------------------------------
# TASK 4
# Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train Linear Regression Model

linear_model = LinearRegression()

linear_model.fit(X_train_reg_scaled, y_reg_train)

# Prediction

y_pred_reg = linear_model.predict(X_test_reg_scaled)

# Evaluation Metrics

mse = mean_squared_error(y_reg_test, y_pred_reg)
r2 = r2_score(y_reg_test, y_pred_reg)

print("="*60)
print("LINEAR REGRESSION RESULTS")
print("="*60)

print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"R² Score                 : {r2:.4f}")

# Feature Coefficients

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

coef_df["Absolute Coefficient"] = coef_df["Coefficient"].abs()

print("\n")

print("="*60)
print("MODEL COEFFICIENTS")
print("="*60)

print(coef_df)

# Top 3 Important Features

top3 = coef_df.sort_values(
    by="Absolute Coefficient",
    ascending=False
).head(3)

print("\n")

print("="*60)
print("TOP 3 FEATURES")
print("="*60)

print(top3)

Output:
============================================================
LINEAR REGRESSION RESULTS
============================================================
Mean Squared Error (MSE) : 35478020.68
R² Score                 : 0.8069


============================================================
MODEL COEFFICIENTS
============================================================
            Feature  Coefficient  Absolute Coefficient
0               age  3472.975553           3472.975553
1               bmi  1927.828251           1927.828251
2          children   636.501185            636.501185
3          sex_male   -50.749675             50.749675
4        smoker_yes  9234.342487           9234.342487
5  region_northwest  -168.944439            168.944439
6  region_southeast  -371.780810            371.780810
7  region_southwest  -284.610396            284.610396


============================================================
TOP 3 FEATURES
============================================================
      Feature  Coefficient  Absolute Coefficient
4  smoker_yes  9234.342487           9234.342487
0         age  3472.975553           3472.975553
1         bmi  1927.828251           1927.828251
--------------------------------------------------------------------------------
# TASK 5
# Ridge Regression Model

from sklearn.linear_model import Ridge

# Train Ridge Regression Model

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train_reg_scaled, y_reg_train)

# Prediction

y_pred_ridge = ridge_model.predict(X_test_reg_scaled)

# Evaluation Metrics

ridge_mse = mean_squared_error(y_reg_test, y_pred_ridge)
ridge_r2 = r2_score(y_reg_test, y_pred_ridge)

print("="*60)
print("RIDGE REGRESSION RESULTS")
print("="*60)

print(f"Mean Squared Error (MSE) : {ridge_mse:.2f}")
print(f"R² Score                 : {ridge_r2:.4f}")

# Comparison Table

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MSE": [mse, ridge_mse],
    "R² Score": [r2, ridge_r2]
})

print("\n")

print("="*60)
print("LINEAR VS RIDGE REGRESSION")
print("="*60)

print(comparison)

# Ridge Coefficients

ridge_coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge_model.coef_
})

ridge_coef["Absolute Coefficient"] = ridge_coef["Coefficient"].abs()

print("\n")

print("="*60)
print("RIDGE REGRESSION COEFFICIENTS")
print("="*60)

print(ridge_coef.sort_values(by="Absolute Coefficient", ascending=False))

Output:
============================================================
RIDGE REGRESSION RESULTS
============================================================
Mean Squared Error (MSE) : 35512474.83
R² Score                 : 0.8067


============================================================
LINEAR VS RIDGE REGRESSION
============================================================
               Model           MSE  R² Score
0  Linear Regression  3.547802e+07  0.806929
1   Ridge Regression  3.551247e+07  0.806741


============================================================
RIDGE REGRESSION COEFFICIENTS
============================================================
            Feature  Coefficient  Absolute Coefficient
4        smoker_yes  9225.380712           9225.380712
0               age  3469.695902           3469.695902
1               bmi  1925.755429           1925.755429
2          children   636.297225            636.297225
6  region_southeast  -369.894956            369.894956
7  region_southwest  -283.873908            283.873908
5  region_northwest  -168.612468            168.612468
3          sex_male   -49.953113             49.953113

--------------------------------------------------------------------------------
# TASK 6
# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Check Class Distribution

print("="*60)
print("CLASS DISTRIBUTION")
print("="*60)

print(y_clf_train.value_counts())

class_ratio = y_clf_train.value_counts(normalize=True)

print("\nClass Ratio")
print(class_ratio)

# Handle Class Imbalance

if class_ratio.min() < 0.35:
    print("\nClass imbalance detected.")
    print("Using class_weight='balanced'.")

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
else:
    print("\nDataset is balanced.")

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

# Train Model

logistic_model.fit(X_train_clf_scaled, y_clf_train)

# Predictions

y_pred_clf = logistic_model.predict(X_test_clf_scaled)

y_pred_prob = logistic_model.predict_proba(X_test_clf_scaled)[:,1]

# Confusion Matrix

cm = confusion_matrix(y_clf_test, y_pred_clf)

print("\n")

print("="*60)
print("CONFUSION MATRIX")
print("="*60)

print(cm)

# Classification Report

print("\n")

print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_clf_test, y_pred_clf))

# ROC Curve and AUC

fpr, tpr, thresholds = roc_curve(y_clf_test, y_pred_prob)

auc = roc_auc_score(y_clf_test, y_pred_prob)

print("\n")

print("="*60)
print("AUC SCORE")
print("="*60)

print("AUC :", round(auc,4))

# Plot ROC Curve

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.show()

Output:
============================================================
CLASS DISTRIBUTION
============================================================
charges
0    535
1    534
Name: count, dtype: int64

Class Ratio
charges
0    0.500468
1    0.499532
Name: proportion, dtype: float64

Dataset is balanced.


============================================================
CONFUSION MATRIX
============================================================
[[120  14]
 [ 11 123]]


============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

           0       0.92      0.90      0.91       134
           1       0.90      0.92      0.91       134

    accuracy                           0.91       268
   macro avg       0.91      0.91      0.91       268
weighted avg       0.91      0.91      0.91       268



============================================================
AUC SCORE
============================================================
AUC : 0.9529

-----------------------------------------------------------------------------
# TASK 7
# Decision Threshold Sensitivity


from sklearn.metrics import precision_score, recall_score, f1_score

print("="*70)
print("DECISION THRESHOLD SENSITIVITY")
print("="*70)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

results = []

for threshold in thresholds:

    # Convert probabilities into class labels
    y_pred_threshold = (y_pred_prob >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_clf_test, y_pred_threshold)
    recall = recall_score(y_clf_test, y_pred_threshold)
    f1 = f1_score(y_clf_test, y_pred_threshold)

    results.append([threshold, precision, recall, f1])

# Create DataFrame
threshold_df = pd.DataFrame(
    results,
    columns=["Threshold", "Precision", "Recall", "F1 Score"]
)

print(threshold_df)

# Best Threshold based on F1 Score

best_row = threshold_df.loc[threshold_df["F1 Score"].idxmax()]

print("\n")

print("="*70)
print("BEST THRESHOLD")
print("="*70)

print(f"Threshold : {best_row['Threshold']}")
print(f"Precision : {best_row['Precision']:.4f}")
print(f"Recall    : {best_row['Recall']:.4f}")
print(f"F1 Score  : {best_row['F1 Score']:.4f}")

Output:
======================================================================
DECISION THRESHOLD SENSITIVITY
======================================================================
   Threshold  Precision    Recall  F1 Score
0        0.3   0.779141  0.947761  0.855219
1        0.4   0.855172  0.925373  0.888889
2        0.5   0.897810  0.917910  0.907749
3        0.6   0.960317  0.902985  0.930769
4        0.7   1.000000  0.843284  0.914980


======================================================================
BEST THRESHOLD
======================================================================
Threshold : 0.6
Precision : 0.9603
Recall    : 0.9030
F1 Score  : 0.9308
-----------------------------------------------------------------------------
# TASK 8
# Logistic Regression with Strong Regularization (C = 0.01)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score
)

# Train Second Logistic Regression Model

logistic_model_c001 = LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model_c001.fit(X_train_clf_scaled, y_clf_train)

# Predictions

y_pred_c001 = logistic_model_c001.predict(X_test_clf_scaled)

y_prob_c001 = logistic_model_c001.predict_proba(X_test_clf_scaled)[:,1]

# Evaluation

precision1 = precision_score(y_clf_test, y_pred_clf)
recall1 = recall_score(y_clf_test, y_pred_clf)
auc1 = roc_auc_score(y_clf_test, y_pred_prob)

precision2 = precision_score(y_clf_test, y_pred_c001)
recall2 = recall_score(y_clf_test, y_pred_c001)
auc2 = roc_auc_score(y_clf_test, y_prob_c001)

# Comparison Table

comparison = pd.DataFrame({

    "Model":[
        "Logistic Regression (C=1.0)",
        "Logistic Regression (C=0.01)"
    ],

    "Precision":[
        precision1,
        precision2
    ],

    "Recall":[
        recall1,
        recall2
    ],

    "AUC":[
        auc1,
        auc2
    ]

})

print("="*70)
print("LOGISTIC REGRESSION COMPARISON")
print("="*70)

print(comparison)

Output:
======================================================================
LOGISTIC REGRESSION COMPARISON
======================================================================
                          Model  Precision    Recall       AUC
0   Logistic Regression (C=1.0)   0.897810  0.917910  0.952885
1  Logistic Regression (C=0.01)   0.884058  0.910448  0.951103

-----------------------------------------------------------------------------
# TASK 9
# Bootstrap Confidence Interval

import numpy as np

auc_difference = []

# Bootstrap Sampling

for i in range(500):

    index = np.random.choice(
        len(y_clf_test),
        size=len(y_clf_test),
        replace=True
    )

    y_true_sample = y_clf_test.iloc[index]

    prob_c1 = y_pred_prob[index]

    prob_c001 = y_prob_c001[index]

    auc_c1 = roc_auc_score(y_true_sample, prob_c1)

    auc_c001 = roc_auc_score(y_true_sample, prob_c001)

    auc_difference.append(auc_c1 - auc_c001)

# Mean Difference
mean_difference = np.mean(auc_difference)

# 95% Confidence Interval
lower = np.percentile(auc_difference,2.5)

upper = np.percentile(auc_difference,97.5)

print("="*70)
print("BOOTSTRAP RESULTS")
print("="*70)

print("Mean AUC Difference :", round(mean_difference,4))

print("95% Confidence Interval")

print("Lower Bound :", round(lower,4))

print("Upper Bound :", round(upper,4))

# Interpretation

if lower > 0 or upper < 0:

    print("\nConfidence interval excludes zero.")
    print("The C=1.0 model consistently outperforms the C=0.01 model.")

else:

    print("\nConfidence interval includes zero.")
    print("The performance difference may not be statistically reliable.")

Output:

======================================================================
BOOTSTRAP RESULTS
======================================================================
Mean AUC Difference : 0.0018
95% Confidence Interval
Lower Bound : -0.0009
Upper Bound : 0.0049

Confidence interval includes zero.
The performance difference may not be statistically reliable.
-----------------------------------------------------------------------------
