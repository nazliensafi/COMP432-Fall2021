# source; https://towardsdatascience.com/machine-learning-classifiers-comparison-with-python-33149aecdbca

# Import required libraries for performance metrics
# Import required libraries for machine learning classifiers
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Define dictionary with performance metrics
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

# Instantiate the machine learning classifiers
logReg_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtc_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
kneighbors_model = KNeighborsClassifier()
adaboost_model = AdaBoostClassifier()
gnb_model = GaussianNB()
neural_model = MLPClassifier()


# Define the models evaluation function
def models_evaluation(X, y, folds):

    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds

    '''

    # Perform cross-validation to each machine learning classifier
    logReg = cross_validate(logReg_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtc = cross_validate(dtc_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    kneighbors = cross_validate(kneighbors_model, X, y, cv=folds, scoring=scoring)
    adaboost = cross_validate(adaboost_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    neural = cross_validate(neural_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models performance metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression': [logReg['test_accuracy'].mean(),
                                                                logReg['test_precision'].mean(),
                                                                logReg['test_recall'].mean(),
                                                                logReg['test_f1_score'].mean()],

                                        'Support Vector Classifier': [svc['test_accuracy'].mean(),
                                                                      svc['test_precision'].mean(),
                                                                      svc['test_recall'].mean(),
                                                                      svc['test_f1_score'].mean()],

                                        'Decision Tree': [dtc['test_accuracy'].mean(),
                                                          dtc['test_precision'].mean(),
                                                          dtc['test_recall'].mean(),
                                                          dtc['test_f1_score'].mean()],

                                        'Random Forest': [rfc['test_accuracy'].mean(),
                                                          rfc['test_precision'].mean(),
                                                          rfc['test_recall'].mean(),
                                                          rfc['test_f1_score'].mean()],

                                        'K neighbors': [kneighbors['test_accuracy'].mean(),
                                                        kneighbors['test_precision'].mean(),
                                                        kneighbors['test_recall'].mean(),
                                                        kneighbors['test_f1_score'].mean()],

                                        'AdaBoost': [adaboost['test_accuracy'].mean(),
                                                     adaboost['test_precision'].mean(),
                                                     adaboost['test_recall'].mean(),
                                                     adaboost['test_f1_score'].mean()],

                                        'Gaussian Naive Bayes': [gnb['test_accuracy'].mean(),
                                                                 gnb['test_precision'].mean(),
                                                                 gnb['test_recall'].mean(),
                                                                 gnb['test_f1_score'].mean()],

                                        'Neural Networks': [neural['test_accuracy'].mean(),
                                                            neural['test_precision'].mean(),
                                                            neural['test_recall'].mean(),
                                                            neural['test_f1_score'].mean()]},

                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)

    # Return models performance metrics scores data frame
    return models_scores_table


# Run models_evaluation function
models_evaluation(df_data, df_target, 5)
