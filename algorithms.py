from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split


def RF(X_train, y_train, X_test, y_test):
    parameters = {'class_weight':['balanced', None],
                'max_depth': [10,20,30,40],
                'max_features': [10,15,20]
                }
    gscv = GridSearchCV(RandomForestClassifier(), parameters)
    fit = gscv.fit(X_train, y_train)
    print('Best parameters for RF: {}'.format(fit.best_params_))
    y_hat_RF = fit.predict_proba(X_test)[:,1]
    y_pred_RF = fit.predict(X_test)
    return y_hat_RF, y_pred_RF, y_test

def GBC(X_train, y_train, X_test, y_test):
    parameters = {'learning_rate':[0.1, 0.5],
                    'n_estimators':  [500,400]
                    }
    decisionTree = GradientBoostingClassifier()
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(X_train, y_train)
    print('Best parameters for GBC: {}'.format(fit.best_params_))
    y_hat_GBC = fit.predict_proba(X_test)[:,1]
    y_pred_GBC = fit.predict(X_test)
    return y_hat_GBC, y_pred_GBC, y_test

def ABC(X_train, y_train, X_test, y_test):
    parameters = {'learning_rate':[0.1,0.5],
                    'n_estimators':  [300,200]
                    }
    decisionTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(X_train, y_train)
    print('Best parameters for ABC: {}'.format(fit.best_params_))
    y_hat_ABC = fit.predict_proba(X_test)[:,1]
    y_pred_ABC = fit.predict(X_test)
    return y_hat_ABC, y_pred_ABC, y_test

def cross_validate(X,y,model):
    # Split into train and test to crossvalidate
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Balance training data
    ads = ADASYN(random_state = 10)
    X_train_b, y_train_b = ads.fit_sample(X_train, y_train)
    if model=='RF':
        return RF(X_train_b, y_train_b, X_test, y_test)
    elif model=='GBC':
        return GBC(X_train_b, y_train_b, X_test, y_test)
    elif model=='ABC':
        return ABC(X_train_b, y_train_b, X_test, y_test)
    else:
        print('Enter a valid model')