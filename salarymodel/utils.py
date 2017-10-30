#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: utils.py
Author: zlamberty
Created: 2017-10-29
"""

import numpy as np
import sklearn.base
import sklearn.preprocessing


class MonetaryLog1P(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, colinds):
        """take the column indices `colinds` of monetary data"""
        self._colinds = colinds

    def fit(self, x):
        return self

    def transform(self, x):
        xc = x.copy()
        xc[:, self._colinds] = np.log1p(xc[:, self._colinds].astype('float'))
        return xc

    def fit_transform(self, x, y=None):
        return self.transform(x)


class MultiColumnLabelEncoder(sklearn.base.BaseEstimator,
                              sklearn.base.TransformerMixin):
    def __init__(self, colinds):
        """take the column indices `colinds` of excpected category features"""
        self._colinds = colinds
        self._les = {
            i: sklearn.preprocessing.LabelEncoder() for i in self._colinds
        }

    def fit(self, x, y=None):
        for (i, lenc) in self._les.items():
            lenc.fit(x[:, i])
        return self

    def transform(self, x):
        xc = x.copy()
        for (i, lenc) in self._les.items():
            xc[:, i] = lenc.transform(xc[:, i])
        return xc

    def fit_transform(self, x, y=None):
        xc = x.copy()
        for (i, lenc) in self._les.items():
            xc[:, i] = lenc.fit_transform(xc[:, i])
        return xc

    def inverse_transforms(self):
        xc = x.copy()
        for (i, lenc) in self._les.items():
            xc[:, i] = lenc.inverse_transform(xc[:, i])
        return xc

    @property
    def classes_(self):
        return {i: lenc.classes_ for (i, lenc) in self._les.items()}
