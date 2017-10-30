#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: salarymodel.py
Author: zlamberty
Created: 2017-10-27
"""

import json
import pickle

import numpy as np

from utils import MonetaryLog1P, MultiColumnLabelEncoder


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

COLUMNS = [
    'age',
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
]


# ----------------------------- #
#   utilities                   #
# ----------------------------- #

def respond(err, res=None):
    return {
        'statusCode': '400' if err else '200',
        'body': json.dumps(str(err)) if err else json.dumps(res),
        'headers': {'Content-Type': 'application/json'},
    }


def build_record(params):
    if params is None:
        raise ValueError("you must provide input record parameters")

    try:
        print('building record from query string paramters')
        return np.array([[params[k] for k in COLUMNS]])
    except KeyError as ke:
        raise KeyError('missing required parameter "{}"'.format(ke))


def load_pipelines():
    # short circuit this by loading the regular pickle files...
    print('loading preprocessor from pkl')
    with open('salary_preprocess_pipeline.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    print('loading modeller from pkl')
    with open('salary_modelling_pipeline.pkl', 'rb') as f:
        modeller = pickle.load(f)

    print('all loaded')
    return preprocessor, modeller


# ----------------------------- #
#   handler function            #
# ----------------------------- #

def handler(event, context):
    print("event = {}".format(event))

    reqtype = event['httpMethod']
    if reqtype == 'GET':
        try:
            record = build_record(event['queryStringParameters'])
            preprocessor, modeller = load_pipelines()
            print('scoring')
            score = modeller.predict_proba(
                preprocessor.transform(record)
            )[0]
            return respond(
                err=None,
                res={'score': dict(zip(['<=50k', '>50k'], score))}
            )
        except Exception as e:
            return respond(e)
    else:
        return respond(ValueError('Unsupported method "{}"'.format(reqtype)))
