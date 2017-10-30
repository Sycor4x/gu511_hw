#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: train.py
Author: zlamberty
Created: 2017-10-29
"""

import pickle

import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from utils import MonetaryLog1P, MultiColumnLabelEncoder


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

# tokens for easier indexing
AGE = 0
WORKCLASS = 1
EDUCATION = 2
MARITAL_STATUS = 3
OCCUPATION = 4
RELATIONSHIP = 5
RACE = 6
SEX = 7
CAPITAL_GAIN = 8
CAPITAL_LOSS = 9
HOURS_PER_WEEK = 10
NATIVE_COUNTRY = 11
TARGET = 12

MONEY_INDICES = [
    CAPITAL_GAIN,
    CAPITAL_LOSS
]

CATEGORY_INDICES = [
    WORKCLASS,
    EDUCATION,
    MARITAL_STATUS,
    OCCUPATION,
    RELATIONSHIP,
    RACE,
    SEX,
    NATIVE_COUNTRY,
]


# ----------------------------- #
#   Main routine                #
# ----------------------------- #

def main():
    """the whole shebang"""
    x, y = load_data()
    preprocess, modelling = build_pipelines()

    # manipulate all input data before splitting for modelling
    x = preprocess.fit_transform(x)
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)

    # split for validation purposes
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, random_state=1337
    )

    modelling.fit(xtrain, ytrain)

    performance_details(modelling, xtest, ytest)

    # finally, save that stuff
    print('pickling preprocess pipeline')
    with open('salary_preprocess_pipeline.pkl', 'wb') as f:
        pickle.dump(preprocess, f)

    print('pickling modelling pipeline')
    with open('salary_modelling_pipeline.pkl', 'wb') as f:
        pickle.dump(modelling, f)


def load_data():
    """dowload and manipulate the input data"""
    columns = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'target'
    ]

    url = (
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
        'adult.data'
    )
    df = pd.read_csv(
        url,
        names=columns,
        delimiter=', ',
        index_col=False,
        engine='python'
    )

    df = df.drop(['fnlwgt', 'education-num'], axis=1)

    x = df.values[:, :-1]
    y = df.values[:, -1]

    return x, y


def build_pipelines():
    """build our two pipelines (a preprocessor and a modeller)"""
    # preprocessing first
    ml1p = MonetaryLog1P(MONEY_INDICES)
    mclenc = MultiColumnLabelEncoder(CATEGORY_INDICES)
    ohenc = sklearn.preprocessing.OneHotEncoder(
        n_values='auto',
        categorical_features=CATEGORY_INDICES,
        sparse=False
    )

    preprocess = sklearn.pipeline.Pipeline(
        steps=[
            # a sequence of name, transformer objects
            ('money_log1p', ml1p),
            ('categorical_encoder', mclenc),
            ('dummy_var_encoder', ohenc),
        ]
    )

    # now modelling in two parts

    # feature selection
    rf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=10,
        n_jobs=-1,
        random_state=1337
    )
    rfe = sklearn.feature_selection.RFE(estimator=rf)

    # modelling
    mrf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=1337,
    )

    modelling = sklearn.pipeline.Pipeline(
        steps=[
            # a sequence of name, transformer objects
            ('rfe', rfe),
            ('random_forest', mrf)
        ]
    )

    return preprocess, modelling


def performance_details(modelling, xtest, ytest):
    """let's just print the confusion matrix and call it good"""
    print(confusion_matrix(modelling, xtest, ytest))


def confusion_matrix(modelling, xtest, ytest, thresh=0.5):
    """create a standard confusion matrix, leveraging some pandas"""
    yproba = modelling.predict_proba(xtest)
    dfpred = pd.DataFrame({
        'y_actual': ytest,
        'y_pred_prob': yproba[:, 1],
        'y_predicted': (yproba[:, 1] >= thresh).astype(int),
    })
    tallcm = dfpred.groupby(['y_actual', 'y_predicted']).count()
    tallcm.columns = ['count']
    return tallcm.unstack()


# ----------------------------- #
#   Command line                #
# ----------------------------- #

if __name__ == '__main__':
    main()
