import pandas as pd
import numpy as np
import pickle
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, accuracy_score
from algorithms import RF, GBC, ABC, cross_validate
from data_cleaning import clean_data

# Load data
df = pd.read_json('data/data.zip')

X = clean_data(df)
y = df.acct_type.map(lambda x: 1 if ((x == 'fraudster') | \
                                    (x == 'fraudster_att') | \
                                    (x == 'fraudster_event')) \
                                    else 0)

# Run Classification Algorithms
y_hat_RF, y_pred_RF, y_test_RF = cross_validate(X,y,'RF')
y_hat_GBC, y_pred_GBC, y_test_GBC = cross_validate(X,y,'GBC')
y_hat_ABC, y_pred_ABC, y_test_ABC = cross_validate(X,y,'ABC')

# Evaluate Performance to pick best model
precision_rf = precision_score(y_test_RF, y_pred_RF)
precision_g = precision_score(y_test_GBC, y_pred_GBC)
precision_a = precision_score(y_test_ABC, y_pred_ABC)
acc_rf = accuracy_score(y_test_RF, y_pred_RF)
acc_g = accuracy_score(y_test_GBC, y_pred_GBC)
acc_a = accuracy_score(y_test_ABC, y_pred_ABC)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_RF, y_hat_RF)
auc_score_rf = auc(fpr_rf, tpr_rf)
fpr_g, tpr_g, thresholds_g = roc_curve(y_test_GBC, y_hat_GBC)
auc_score_g = auc(fpr_g, tpr_g)
fpr_a, tpr_a, thresholds_a = roc_curve(y_test_ABC, y_hat_ABC)
auc_score_a = auc(fpr_a, tpr_a)

print('Precision')
print('RF: ', precision_rf, '\nGB: ', precision_g, '\nAB: ', precision_a)

print('Recall')
print('RF: ', acc_rf, '\nGB: ', acc_g, '\nAB: ', acc_a)

print('AUC')
print('RF: ', auc_score_rf, '\nGB: ', auc_score_g, '\nAB: ', auc_score_a)

# Creating Pickle file
model = AdaBoostClassifier(learning_rate=0.5, n_estimators=200)
model.fit(X,y)
# model.predict()
pickle.dump(model,open('model.p','wb'))