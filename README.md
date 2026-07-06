# Capstone2

Part 2 – Supervised Machine Learning Model
Dataset
The dataset used is the Medical Cost Personal Dataset. It contains information about individual medical insurance charges based on personal attributes such as age, sex, BMI, number of children, smoking status, and region.

The cleaned dataset (cleaned_data.csv) generated in Part 1 was used for this project.

Target Variables
Regression Label (y_reg)
The regression target is the charges column, which represents the medical insurance cost for each individual.

y_reg = df["charges"]
Classification Label (y_clf)
A binary classification label was created by comparing each insurance charge with the median charge.

y_clf = (df["charges"] > df["charges"].median()).astype(int)
1 = Charges greater than the median
0 = Charges less than or equal to the median
Data Preprocessing
Encoding
The dataset contains three categorical columns:

sex
smoker
region
These columns have no natural order, so One-Hot Encoding was applied using:

pd.get_dummies(drop_first=True)
One-Hot Encoding prevents introducing a false ordinal relationship between categories. The first dummy column was dropped to avoid multicollinearity.

Train-Test Split
The dataset was divided into:

80% Training Data
20% Testing Data
using:

train_test_split(test_size=0.2, random_state=42)
Feature Scaling
StandardScaler was applied only to the training data and then used to transform the testing data.

This prevents data leakage because fitting the scaler on the full dataset would allow information from the test data to influence the training process.

Linear Regression
The Linear Regression model was trained using the scaled training data.

Evaluation metrics:

Mean Squared Error (MSE)
R² Score
The model coefficients were examined to identify the three most influential features.

Interpretation
A large positive coefficient means that increasing the standardized feature by one unit increases the predicted insurance charge by the coefficient value.

A large negative coefficient means that increasing the standardized feature decreases the predicted insurance charge.

Ridge Regression
Ridge Regression was trained with:

alpha = 1.0
Its MSE and R² Score were compared with the standard Linear Regression model.

Why Ridge Regression?
Ridge Regression reduces overfitting by shrinking large coefficients using L2 regularization.

The alpha parameter controls the strength of regularization.

Small alpha → behaves like Linear Regression
Large alpha → stronger regularization
Classification Model
A Logistic Regression classifier was trained to predict whether a person's insurance charge is above the median.

If class imbalance existed, the model used:

class_weight="balanced"
This adjusts class weights automatically during training.

Evaluation Metrics
The following metrics were computed:

Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC Curve
AUC Score
Precision and Recall
Precision Formula

Precision = TP / (TP + FP)

Recall Formula

Recall = TP / (TP + FN)

Where:

TP = True Positive
FP = False Positive
FN = False Negative
For predicting high medical insurance charges, Recall is more important because failing to identify high-cost individuals may reduce the usefulness of the prediction model.

ROC Curve and AUC
The ROC Curve illustrates the trade-off between True Positive Rate and False Positive Rate.

The AUC value measures the classifier's ability to distinguish between the two classes.

AUC = 1.0 → Perfect classifier
AUC = 0.5 → Random guessing
Higher AUC values indicate better classification performance.

Decision Threshold Sensitivity
The Logistic Regression model was evaluated at thresholds:

0.30
0.40
0.50
0.60
0.70
For each threshold:

Precision
Recall
F1 Score
were calculated.

The threshold with the highest F1 Score was selected as the best balance between Precision and Recall.

Increasing the threshold generally increases Precision but decreases Recall.

Lowering the threshold generally increases Recall but decreases Precision.

Logistic Regression Regularization
A second Logistic Regression model was trained using:

C = 0.01
The performance was compared with the baseline model (C = 1.0).

The parameter C controls regularization strength.

Smaller C → Stronger Regularization
Larger C → Weaker Regularization
Strong regularization reduces overfitting but may also reduce predictive performance.

Bootstrap Confidence Interval
A bootstrap experiment with 500 resamples was performed.

For each bootstrap sample:

AUC(C=1.0)
AUC(C=0.01)
were calculated.

The difference between the two AUC values was recorded.

Finally, the following statistics were computed:

Mean AUC Difference
2.5th Percentile
97.5th Percentile
If the confidence interval excludes zero, the C=1.0 model consistently performs better.

If the confidence interval includes zero, the difference may not be statistically significant.

Conclusion
Two supervised learning models were developed:

Linear Regression for predicting medical insurance charges.
Logistic Regression for classifying insurance charges as above or below the median.
Both regression and classification models were evaluated using appropriate performance metrics. Regularization and bootstrap analysis were also performed to improve model reliability and assess performance consistency.
