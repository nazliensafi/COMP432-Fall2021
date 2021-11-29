##Sources
# https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

#Decisions:
#Imputer (N) vs dropna (Y)
#Kfold (N)
#Use column names or not (+ handling of unknown columns)

import numpy as np
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def load_data():
    filenames = {
        'train': 'Classification\\Datasets\\Adult\\adult.data',
        'test': 'Classification\\Datasets\\Adult\\adult.test'
        }
    
    train = pd.read_csv(filenames['train'], comment='|', header=None, delimiter=',', dtype=None, na_values='?')
    train.dropna()
    X_train = train.values[:,:-1]
    y_train = train.values[:,-1]

    test = pd.read_csv(filenames['test'], comment='|',  header=None, delimiter=',', dtype=None, na_values='?')
    X_test = test.values[:,:-1]
    y_test = test.values[:,-1]

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    cat_enc = OneHotEncoder(handle_unknown='ignore')
    num_enc = StandardScaler()
    label_enc = LabelEncoder()

    cat_features = X_train.select_dtypes(include=['object']).columns
    num_features = X_train.select_dtypes(include=['int64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_enc, num_features),
            ("cat", cat_enc, cat_features)
        ]
    )

def main():
    X_train, y_train, X_test, y_test = load_data()
    preprocess_data(X_train, y_train, X_test, y_test)


    # clf = Pipeline(
    #     steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
    # )

    #clf.fit(X_train, y_train)


if __name__ == "__main__":
    main()