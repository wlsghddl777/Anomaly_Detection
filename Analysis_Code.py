# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 모듈 import

# +
import pickle
import numpy as np
import pandas as pd


import easydict


#시각화 모듈
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
import seaborn as sns




import scipy as sp
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, recall_score, precision_score,f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import pandas as pd
from datetime import datetime, date

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, recall_score, precision_score,f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import statsmodels.api as sm

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from tensorflow.keras import activations
import itertools
import time
import random as rn
RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(
    RANDOM_SEED
)

from sklearn.neighbors import KNeighborsRegressor
from math import sqrt


import matplotlib

import warnings
warnings.filterwarnings('ignore')

# 한글폰트 사용시 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


# -

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


# # 데이터 EDA

trnd_data = pd.read_csv("TH_CM_FMCS_TRND_DATA.csv")
A_5차 = load_pickle('A_5차.pickle')
B_5차 = load_pickle('B_5차.pickle')
ETC_5차 = load_pickle('ETC_5차.pickle')
설비번호맵핑 = pd.read_excel("설비번호맵핑.xlsx")
설비번호맵핑 = 설비번호맵핑[['EQP_NO','ORDER']]
grade_B_alrm = pd.read_excel('B등급 알람센서.xlsx', sheet_name='Sheet2')

trnd_data


