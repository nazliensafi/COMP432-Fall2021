##Sources
# https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

#Decisions:
#Imputer (N) vs dropna (Y)
#Kfold (N)
#Use column names or not (+ handling of unknown columns)

import numpy as np
import sklearn as sk
from sklearn import linear_model, tree, svm, ensemble, neighbors, naive_bayes, neural_network
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import pandas as pd

# Relative path root for datasets
file_loc = 'Classification\\Datasets\\'

'''
Dataset metadata, including file name, params to send to the function reading the file, and cost matrices if applicable
'''
dataset_details = {
    'credit':
        {
            'file': ['default of credit card clients.xls'],
            'load_params': {
                'index_col': 0,
                'skiprows': [0,1]
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
        'file': ['ThoracicSurgery.arff'],
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

# Classifier metadata, including classifier and hyperparameters once chosen
CLASSIFIERS = {
    'logreg': {
        'clf': linear_model.LogisticRegression,
        'params': {}
    },
    'svc': {
        'clf': svm.SVC,
        'params': {}
    },
    'tree': {
        'clf': tree.DecisionTreeClassifier,
        'params': {}
    },
    'forest': {
        'clf': ensemble.RandomForestClassifier,
        'params': {}
    },
    'kneighbors': {
        'clf': neighbors.KNeighborsClassifier,
        'params': {}
    },
    'adaboost': {
        'clf': ensemble.AdaBoostClassifier,
        'params': {}
    },
    'nb': {
        'clf': naive_bayes.GaussianNB,
        'params': {}
    },
    'neural': {
        'clf': neural_network.MLPClassifier,
        'params': {}
    }
}


def load_dataset(dataset, file_loc): # passingdataset_details instead of dataset

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
    
    X = df.values[:,:-1]
    y = df.values[:,-1]

    return X, y


def create_encoders(X, y):

    '''
    Splits dataset into numerical and categorical data
    creates relevant encoders for both features and labels
    '''

    cat_enc = OneHotEncoder(handle_unknown='ignore')
    num_enc = StandardScaler()

    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['int64', 'float32']).columns

    X_enc = ColumnTransformer(
        transformers=[
            ("num", num_enc, num_features),
            ("cat", cat_enc, cat_features)
        ]
    )

    y_enc = None

    if y.dtypes=='int64' or y.dtypes=='float32':
        y_enc = LabelEncoder(y).fit

    return X_enc, y_enc


def preprocessor(dataset_details):

    '''
    Loads encoded datasets in
    standardizes to dataframe or np array
    splits into train and testing data
    '''

    train_data = {}
    test_data = {}
    for dataset in dataset_details:
        X, y = load_dataset(dataset_details[dataset], file_loc) #this is being done in get_X_and_y(dataset_details): line 119
        X_enc, y_enc = create_encoders(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, random_state=0)
        train_data[dataset] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_enc': X_enc,
            'y_enc': y_enc
        }
        test_data[dataset] = {
            'X_test': X_test,
            'y_test': y_test,
            'X_enc': X_enc,
            'y_enc': y_enc
        }
    return train_data, test_data


def load_excel(file,  **kwargs):
    df = pd.read_excel(file, dtype=None, engine='xlrd', **kwargs)

    return df


def load_arff(file):
    df = loadarff(file)
    df = pd.DataFrame(df, dtype=None)

    return df


def load_plaintext(file, **kwargs):
    df = pd.read_csv(file, header=None, dtype=None, **kwargs)
    df.dropna()

    return df


def train_classifiers(data, CLASSIFIERS):

    '''
    Trains every classifier on every dataset
    '''

    models = {}
    for clf in CLASSIFIERS:
        for dataset in data:
            model = train_clf(CLASSIFIERS[clf], data[dataset]['X_train'], data[dataset]['y_train'])
            models[clf][dataset] = model

    return models


def train_clf(clf_data, X_train, y_train):

    '''
    Trains a given classifier on a given dataset
    '''

    X_enc, y_enc = create_encoders(X_train, y_train)
    X = X_enc.fit_transform(X_train)
    if y_enc:
        y = y_enc.fit_transform(y_train)
    clf = clf_data['clf']
    load_params = clf_data['load_params']

    model = clf.fit(X_train,y_train, **load_params)

    return model


def main():
    # X, y = load_dataset(dataset_details, file_loc)
    # X_enc, y_enc = create_encoders(X, y)
    train_data, test_data = preprocessor(dataset_details)
    models = train_classifiers(train_data, CLASSIFIERS)


if __name__ == "__main__":
    main()
