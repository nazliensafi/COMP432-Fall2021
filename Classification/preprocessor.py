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

file_loc = 'Classification\\Datasets\\'

DATASET_DETAILS = {
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
CLASSIFIERS = {
    'logreg': {
        'clf': linear_model.LogisticRegression, 
        'load_params': {}
    },
    'svc': {
        'clf': svm.SVC, 
        'load_params': {}
    },
    'tree': {
        'clf': tree.DecisionTreeClassifier,
        'load_params': {}
    },
    'forest': {
        'clf': ensemble.RandomForestClassifier,
        'load_params': {}
    },        
    'kneighbors': {
        'clf': neighbors.KNeighborsClassifier,
        'load_params': {}
    },
    'adaboost': {
        'clf': ensemble.AdaBoostClassifier,
        'load_params': {}
    },
    'nb': {
        'clf': naive_bayes.GaussianNB,
        'load_params': {}
    },
    'neural': {
        'clf': neural_network.MLPClassifier,
        'load_params': {}
    }
}

def load_dataset(dataset, file_loc):
    '''
    Split:
        If split, read test and train separately (or combine?)
    Types: (pandas.read_csv vs pandas.read_excel)
        .xls
        .data / .data-numeric
        .arff
    if not xls:
        Comments:
            If read_csv: comments=dataset['comments']
        Index_col:
            If index_col: index_col=dateset['index_col'] 
        Missing_values:
            If missing_values: na_filter=dataset['missing_values']
    Data:
        if data = Mixed: split and call OneHotEncoder + StandardScaler
        If data = Mumeric: call StandardScaler
    Label:
        if label=categorical: label=LabelEncoder(label)
    *Weighted:
        if weighted: use cost_matrix (?)
    '''
    metadata = DATASET_DETAILS[dataset]
    filenames = metadata['file']
    load_params = metadata['load_params']

    extension = filenames[0].split('.')[1] #Get file type - ASSUMPTION: If split into test and train, both are the same type
    if extension == 'xls': 
        df = load_excel(filenames, file_loc, **load_params)
    elif extension == 'arff':
        df = load_arff(filenames, file_loc)
    else:
        df = load_plaintext(file_loc, filenames, **load_params)

    X = df.values[:,:-1]
    y= df.values[:,-1]
    print(X)
    return X, y

def load_excel(filenames, file_loc, **kwargs):

    df = pd.read_excel(
        f'{file_loc}{filenames[0]}', dtype=None, engine='xlrd', **kwargs
        )

    if len(filenames)>1: #If test and train are split, append them for preprocessing
        test = pd.read_csv(
            f'{file_loc}{filenames[1]}', dtype=None, engine='xlrd', **kwargs
            )
        df = pd.concat([df,test])

    return df

def load_arff(filenames, file_loc):
    df = loadarff(f'{file_loc}{filenames[0]}')[0]
    df = pd.DataFrame(df, dtype=None)
    if len(filenames)>1:
        test = loadarff(f'{file_loc}{filenames[1]}')[0]
        df = np.vstack([df,test])

    return df

def load_plaintext(file_loc, filenames, **kwargs):

    df = pd.read_csv(
        f'{file_loc}{filenames[0]}', header=None, dtype=None, **kwargs
        )

    if len(filenames)>1: #If test and train are split, append them for preprocessing
        test = pd.read_csv(
            f'{file_loc}{filenames[1]}', header=None, dtype=None, **kwargs
            )
        df = pd.concat([df,test])

    df.dropna()

    return df

def create_encoders(X, y):
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

def train_classifiers(data, CLASSIFIERS):
    models = {}
    for clf in CLASSIFIERS:
        for dataset in data:
            model = train_clf(CLASSIFIERS[clf], data[dataset]['X_train'], data[dataset]['y_train'])
            models[clf][dataset] = model
    
    return models

def train_clf(clf_data, X_train, y_train):
    X_enc, y_enc = create_encoders(X_train, y_train)
    X = X_enc.fit_transform(X_train)
    if y_enc:
        y = y_enc.fit_transform(y_train)
    clf = clf_data['clf']
    load_params = clf_data['load_params']

    model = clf.fit(X_train,y_train, **load_params)

    return model

def preprocessor(DATASET_DETAILS):
    train_data = {}
    test_data = {}
    for dataset in DATASET_DETAILS:
        X, y = load_dataset(dataset, file_loc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        train_data[dataset] = {
            'X_train': X_train,
            'y_train': y_train,
        }
        test_data[dataset] = {
            'X_test': X_test,
            'y_test': y_test
        }
    return train_data, test_data

def main():
    train_data, test_data = preprocessor(DATASET_DETAILS)
    models = train_classifiers(train_data, CLASSIFIERS)
    print("done")

if __name__ == "__main__":
    main()