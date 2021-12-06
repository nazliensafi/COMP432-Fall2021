# 1. Default of credit card clients:

import numpy as np
import os
import sklearn as sk
from sklearn import linear_model, tree, svm, ensemble, neighbors, naive_bayes, neural_network
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from scipy.io.arff import loadarff
import pandas as pd
import scipy
### Data and Classifier Parameters  
#Relative path root for datasets
file_loc = os.path.join(os.getcwd(),'Classification/Datasets/')

#Dataset files and parameters based on descriptions
dataset_details = {
    'credit':
        {
            'file': ['default of credit card clients.xls'],
            'load_params': {
                'index_col': 0,
                'skiprows': 1
            }
        },
    'breast_cancer':
        {
            'file': ['breast-cancer-wisconsin.data'],
            'load_params': {
                'na_values': '?',
                'index_col': 0
            },
            'weighted': 'Y'
        },
    'statlog':
        {
            'file': ['german.data-numeric'],
            'load_params': {
                'delim_whitespace': 'true'
            },
            'weighted': 'Y',
        },
    'adult':
        {
            'file': ['adult.data','adult.test'],
            'load_params': {
                'na_values': '?',
                'comment': '|'
            }
        },
    'yeast': {
        'file': ['yeast.data'],
        'load_params': {
            'index_col': 0,
            'delim_whitespace': 'true'
        }
    },
    'thoracic': {
        'file': ['ThoraricSurgery.arff'],
        'load_params': {}
    },
    'seismic': {
        'file': ['seismic-bumps.arff'],
        'load_params': {}
    },
    'retinopathy': {
        'file': ['messidor_features.arff'],
        'load_params': {
            'comment': '@'
        }
    }
}

# Classifiers used - params will be filled in with the chosen hyperparameters
# Classifiers used and their subset of hyperparameters chosen to test with gridsearch
CLASSIFIERS = {
    'logreg': {
        'clf': linear_model.LogisticRegression,
        'param_grid': {
            'penalty' : ['l1', 'l2'],
            'C' : [np.logspace(-4, 4, 20)]
        },
        'params': {
            'random_state': 0
        }
    },
    'tree': {
        'clf': tree.DecisionTreeClassifier,
        'param_grid': {
            'criterion': ['gini','entropy'],
            'min_samples_split': [5,10,50,100],
            'min_samples_leaf': [1, 2, 4],
            'max_depth': [1,5,10,50,100,None]
        },
        'params': {
            'random_state': 0
        }
    },
    'kneighbors': {
        'clf': neighbors.KNeighborsClassifier,
        'param_grid': {
            'leaf_size': list(range(1,50)),
            'n_neighbors': list(range(1,30)),
            'p': [1,2]
        },
        'params': {}
    },
    'adaboost': {
        'clf': ensemble.AdaBoostClassifier,
        'param_grid': {
            'n_estimators': [10, 50, 100, 500],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
            },
        'params': {
            'random_state': 0
        }
    },
    'nb': {
        'clf': naive_bayes.GaussianNB,
        'param_grid': {
            'var_smoothing': [np.logspace(0,-9, num=100)]
        },        
        'params': {}
    }
}
### Preprocessing the data
def preprocessor(dataset_details, file_loc):

    '''
    Loads all datasets in
    Standardizes to dataframe
    Splits into train and testing data
    '''

    train_data = {}
    test_data = {}

    for dataset in dataset_details:
        X, y = load_dataset(dataset_details[dataset], file_loc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        if dataset=='breast_cancer': #train_test_split is converting all 10.0 values to NaN for this dataset for some reason
            X_train = X_train.fillna(value=10)
            X_test = X_test.fillna(value=10)

        train_data[dataset] = {
            'X_train': X_train,
            'y_train': y_train
        }
        test_data[dataset] = {
            'X_test': X_test,
            'y_test': y_test
        }

    return train_data, test_data

def load_dataset(dataset, file_loc):
    '''
    Loads in a dataset according to type and load_params
    Assumes dataset file is either .xls, .arff, or plain text
    If test and train are pre-split, assumes they are the same file type and combines for preprocessing.
    Separates out the last column as y
    '''

    #metadata = dataset_details[dataset] # using dataset_details directly rather than metadata``
    filenames = dataset['file']
    load_params = dataset['load_params']

    dfs = []
    for file in filenames:
        extension = file.split('.')[1]  # Get file type
        file = f'{file_loc}{file}'
        if extension == 'xls':
            df = load_excel(file, **load_params)
        elif extension == 'arff':
            df = load_arff(file)
        else:
            df = load_plaintext(file, **load_params)
        dfs.append(df)
        df = pd.concat(dfs)
    
    y = df.iloc[:,-1]
    X = df = df.iloc[: , :-1]

    return X, y

def load_excel(file,  **kwargs):
    df = pd.read_excel(file, dtype=None, engine='xlrd', **kwargs)
    return df

def load_arff(file):
    data = loadarff(file)
    df = pd.DataFrame(data[0])
    df.dropna()
    return df

def load_plaintext(file, **kwargs):
    df = pd.read_csv(file, header=None, dtype=None, **kwargs)
    df.dropna()
    return df

### Train Data
def train_classifiers(data, CLASSIFIERS):

    '''
    Trains every classifier on every dataset
    '''

    models = {}
    for clf in CLASSIFIERS:
        models[clf]={}
        for dataset in data:
            print("Training ",clf," on ",dataset)
            model, X_enc, y_enc, X_imp = train_clf(CLASSIFIERS[clf], data[dataset]['X_train'], data[dataset]['y_train'])
            models[clf][dataset] = {
                'model': model,
                'X_enc': X_enc,
                'y_enc': y_enc,
                'X_imp': X_imp
            }
        

    return models

def train_clf(clf_data, X, y):

    '''
    Trains a given classifier on a given dataset
    '''
    X_enc, y_enc = encode_or_scale(X, y)
    X = X_enc.transform(X)
    
    if scipy.sparse.issparse(X):
        X = X.toarray()
    if y_enc:
        y = y_enc.transform(y)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imp = imp.fit(X)
    X = X_imp.transform(X)
    params = clf_data['params']
    clf = clf_data['clf']
    model = clf(**params).fit(X,y)
    
    return model, X_enc, y_enc, X_imp

def encode_or_scale(X, y):

    '''
    Splits dataset into numerical and categorical data
    creates relevant encoders for both features and labels
    '''

    cat_enc = OneHotEncoder(handle_unknown='ignore')
    num_enc = StandardScaler()

    cat_features = X.select_dtypes(include=['object']).columns
    num_features =X.select_dtypes(include=['int64', 'float64']).columns

    if len(cat_features)==0:
        X_enc = num_enc
    elif len(num_features)==0:
        X_enc = cat_enc
    else:
        X_enc = ColumnTransformer(
            transformers=[
                ("num", num_enc, num_features),
                ("cat", cat_enc, cat_features)
            ]
        )

    X_enc.fit(X)
    
    y_enc = None
    if y.dtypes=='object':
        y_enc = LabelEncoder().fit(y)


    return X_enc, y_enc
    
### Test Models
def test_classifiers(data, models):
    for clf in models:
        for dataset in models[clf]:
            test_clf(models[clf][dataset], data[dataset]['X_test'], data[dataset]['y_test'])

def test_clf(models, X, y):
    model = models['model']
    X = models['X_enc'].transform(X)
    
    if scipy.sparse.issparse(X):
        X = X.toarray()
    if models.get('y_enc'):
        y = models['y_enc'].transform(y)
    
    print(model)
    print(model.score(X,y))
    
def find_hyperparams(model, classifier_details, X, y):

    param_grid = classifier_details['param_grid']
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)

    result = search.fit(X, y)

    return result.best_params_

def find_all_hyperparams(data, models, classifiers):
    for clf in models:
        for dataset in models[clf]:
            hps = find_hyperparams(models[clf][dataset], classifiers[clf], data[dataset]['X_train'], data[dataset]['y_train'])
            classifiers[clf]['final_params'] = hps


train_data, test_data = preprocessor(dataset_details, file_loc)
models = train_classifiers(train_data, CLASSIFIERS)
find_all_hyperparams(train_data,models,CLASSIFIERS)
test_classifiers(test_data, models)









