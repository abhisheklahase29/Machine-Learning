# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:43:21 2024

@author: abhil
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hiring.csv")

df.head(4)

df.isnull().sum()


df.experience  = df.experience.fillna("One")
df.experience
from word2number import w2n
df
df.experience = df.experience.apply(w2n.word_to_num)
df

df.test_score.mean()
import math
test_score_mean = math.floor(df.test_score.mean())
test_score_mean

df.test_score = df.test_score.fillna(test_score_mean)
df


from sklearn import linear_model
LR = linear_model.LinearRegression()
LR.fit(df[['experience','test_score','interview_score']],df.salary)

LR.predict([[15,10,10]])
