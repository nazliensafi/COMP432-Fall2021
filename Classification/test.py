# 1. Default of credit card clients:

import numpy as np
import os
import sklearn as sk
from sklearn import linear_model, tree, svm, ensemble, neighbors, naive_bayes, neural_network
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from scipy.io.arff import loadarff
import pandas as pd
import scipy
import matplotlib.pyplot as plt
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
    # 'adult':
    #     {
    #         'file': ['adult.data','adult.test'],
    #         'load_params': {
    #             'na_values': '?',
    #             'comment': '|'
    #         }
    #     },
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
    'neural': {
        'clf': neural_network.MLPClassifier,
        'param_grid': {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'max_iter': [500]
        },
        'params': {
            'max_iter': 500
        }
    }
}
### Preprocessing the data
def preprocessor(dataset_details, file_loc, strategy='mean', encoder=StandardScaler, gridsearchflag=False):

    '''
    Loads all datasets in
    Standardizes to dataframe
    Splits into train and testing data
    '''

    train_data = {}
    test_data = {}

    for dataset in dataset_details:
        if (gridsearchflag==True):
            if (dataset!='yeast'):
                continue
        X, y = load_dataset(dataset_details[dataset], file_loc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        #Encode categorical values and scale numeric values
        X_enc, y_enc = create_encoders(X_train, y_train, encoder)
        X_train = X_enc.transform(X_train)
        X_test = X_enc.transform(X_test)
        
        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        if y_enc:
            y_train = y_enc.transform(y_train)
            y_test = y_enc.transform(y_test)

        X_imp, X_train, X_test = impute(X_train, X_test, strategy)

        train_data[dataset] = {
            'X_train': X_train,
            'y_train': y_train,
        }
        test_data[dataset] = {
            'X_test': X_test,
            'y_test': y_test,
            'X_imp': X_imp,
            'X_enc': X_enc,
            'y_enc': y_enc
        }

    return train_data, test_data

def impute(X_train, X_test, strategy='mean'):
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X_imp = imp.fit(X_train)
    X_train = X_imp.transform(X_train)
    X_test = X_imp.transform(X_test)

    return X_imp, X_train, X_test

def create_encoders(X, y, encoder):

    '''
    Splits dataset into numerical and categorical data
    creates relevant encoders for both features and labels
    '''

    cat_enc = OneHotEncoder(handle_unknown='ignore')
    num_enc = encoder()

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
    
def load_dataset(dataset, file_loc):
    '''
    Loads in a dataset according to type and load_params
    Assumes dataset file is either .xls, .arff, or plain text
    If test and train are pre-split, assumes they are the same file type and combines for preprocessing.
    Separates out the last column as y
    '''

    #metadata = dataset_details[dataset]
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
    return df

def load_plaintext(file, **kwargs):
    df = pd.read_csv(file, header=None, dtype=None, **kwargs)
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
            model = train_clf(CLASSIFIERS[clf], data[dataset]['X_train'], data[dataset]['y_train'])
            models[clf][dataset] = {
                'model': model
            }
        

    return models

def train_clf(clf_data, X, y):

    '''
    Trains a given classifier on a given dataset
    '''
    params = clf_data['params']
    clf = clf_data['clf']
    model = clf(**params).fit(X,y)
    
    return model
    
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
    
def find_hyperparams(classifier_details, X, y, gridsearch):
    params = classifier_details['params']
    model = classifier_details['clf'](**params)
    param_grid = classifier_details['param_grid']
    cv = KFold(n_splits=5)
    search = gridsearch(model, param_grid, cv=cv, n_jobs=-1)
    result = search.fit(X, y)
    print (result.best_params_)

def find_all_hyperparams(data, classifiers, datasets, gridsearch=GridSearchCV, gridsearchflag=False):
    count = 0
    for clf in classifiers:
        for dataset in datasets:
            if (gridsearchflag==True):
                if (dataset!='yeast'): continue
            find_hyperparams(classifiers[clf], data[dataset]['X_train'], data[dataset]['y_train'], gridsearch)
        count+=1



#train_data, test_data = preprocessor(dataset_details, file_loc)
#models = train_classifiers(train_data, CLASSIFIERS)
#find_all_hyperparams(train_data,CLASSIFIERS, dataset_details)
#test_classifiers(test_data, models)

print("logreg",sum([0.81,0.97,0.75,0.57,0.60,0.82,0.94,0.67])/8)
print("tree",sum([0.82,0.96,0.72,0.57,0.59,0.82,0.94,0.63])/8)
print("kneighbors",sum([0.81,0.97,0.72,0.53,0.58,0.82,0.95,0.62])/8)
print("adaboost",sum([0.83,0.96,0.74,0.54,0.44,0.82,0.95,0.66])/8)
print("forest",sum([0.82,0.97,0.76,0.58,0.63,0.82,0.95,0.69])/8)
print("nb",sum([0.64,0.95,0.69,0.19,0.11,0.19,0.42,0.61])/8)
print("svc",sum([0.82,0.96,0.75,0.57,0.61,0.82,0.95,0.66])/8)
print("neural",sum([0.81,0.97,0.75,0.52,0.60,0.80,0.94,0.70])/8)






