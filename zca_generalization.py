import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
# import tensorflow_datasets as tfds
from tensorflow import keras
# from tensorflow.keras import layers
import tensorflow_addons as tfa
import h5py
import numpy as np
import math
import os
import random
import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X):
        """Compute the mean, whitening and dewhitening matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        """
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, S, _ = linalg.svd(cov)
        s = np.sqrt(S.clip(self.regularization))
        s_inv = np.diag(1./s)
        s = np.diag(s)
        self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
        self.dewhiten_ = np.dot(np.dot(U, s), U.T)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whitening
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_) + self.mean_

conditions=['S1','S4','S3']
fault = 'A'

data_list = []
Generated=False

for condition in conditions:
# 加载ZCA数据(训练)
    with open('../Data/features/Cross/Cross_'+condition+'/cross_' + fault+'-'+condition+'-L5-01_train.pkl', 'rb') as f:
        train_zca = np.array(pickle.load(f))

    # 加载ZCA数据(测试)
    with open('../Data/features/Cross/Cross_'+condition+'/cross_' + fault+'-'+condition+'-L5-01_test.pkl', 'rb') as f:
        test_zca = np.array(pickle.load(f))

    x_zca = np.vstack((train_zca, test_zca))

    data_list.extend(x_zca)

data_list = np.array(data_list)
trf=ZCA().fit(data_list)

conditions=['S2']
for condition in conditions:
    with open('../Data/features/Cross/Cross_'+condition+'/cross_' + fault+'-'+condition+'-L5-01_train.pkl', 'rb') as f:
        train_zca = np.array(pickle.load(f))

    # 加载ZCA数据(测试)
    with open('../Data/features/Cross/Cross_' + condition + '/cross_' + fault + '-' + condition + '-L5-01_test.pkl',
              'rb') as f:
        test_zca = np.array(pickle.load(f))

    x_hat_train = trf.inverse_transform(train_zca)
    x_hat_test = trf.inverse_transform(test_zca)

x = np.vstack((x_hat_train, x_hat_test))

print(fault,condition)
# # 保存一维数组
with open('../TL-Fault-Diagnosis-Library-main/datasets/GearFace20/condition_1/Gen_' + fault + '/vec_' + fault + '-' + condition + '-L5-01.pkl','wb') as f:
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)






