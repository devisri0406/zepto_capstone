# Modeling Report

## Class balance

survived
0    549
1    342



A stratified split was used because the target contains two classes and stratification preserves their observed proportions in train and test.


## Classification comparison

              model  accuracy  precision   recall       f1      auc     confusion_matrix
Logistic Regression  0.804469   0.793103 0.666667 0.724409 0.843742 [[98, 12], [23, 46]]
      Decision Tree  0.765363   0.754717 0.579710 0.655738 0.797101 [[97, 13], [29, 40]]
      Random Forest  0.815642   0.800000 0.695652 0.744186 0.830040 [[98, 12], [21, 48]]

## Imbalance comparison

                 variant  precision   recall       f1
                baseline   0.800000 0.695652 0.744186
   class_weight_balanced   0.750000 0.739130 0.744526
SMOTE_training_fold_only   0.761194 0.739130 0.750000

SMOTE was applied inside an imbalanced-learn pipeline after the train/test split, so oversampling is limited to the training fold.


## GridSearchCV

Best parameters: {'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}

OOB score: 0.8272

Tuned test metrics: {'accuracy': 0.8156424581005587, 'precision': 0.875, 'recall': 0.6086956521739131, 'f1': 0.717948717948718, 'auc': 0.8431488801054018}


## Regression

MAE: 19.7820

RMSE: 30.7626

R2: 0.3884

Adjusted R2: 0.3744


Heteroscedasticity conclusion: inspect the residual plot. A widening or narrowing residual spread across fitted values indicates heteroscedasticity; a roughly constant random spread does not.


## Reload check

Reloaded pipeline predictions on raw rows: [0, 0, 0]


## Metric-group structure

Classification metrics (accuracy, precision, recall, F1, AUC) are kept separate from regression metrics (MAE, RMSE, R2, Adjusted R2) because the two model types use different scales and objectives.


## Final classifier recommendation

Random Forest is selected for deployment because it achieved the highest test accuracy (0.8156) and F1 score (0.7442) among the three evaluated classifiers. It also achieved a precision of 0.8000, recall of 0.6957, and AUC of 0.8300. The tuned Random Forest achieved an OOB score of 0.8272 and a test accuracy of 0.8156, providing a consistent result on the held-out test set.
