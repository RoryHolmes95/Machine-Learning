import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sb
import sklearn as sk

from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

def avg_approx(row, data, group, fill, avgs):
    x = (data[group].unique())
    if pd.isnull(fill):
        for i in x:
            if group == i:
                return avgs[i]
    else:
        return fill

def multLogReg():

    '''Performs custom Multiple Logarithmic Regression on a dataset of the
       users choosing'''
    address = input("Copy and paste your csv file here")
    data = pd.read_csv(address)
    cols = data.columns
    columns = []
    for i in (cols):
        columns.append(i)
    num = int(input(f"how many of the following fields have binary answers? {columns}"))
    lst = []
    for i in range(num):
        lst.append(0)
    for i in range(num):
        lst[i] = input(f"one at a time, select those columns from the following: {columns} ")
    for i in lst:
        if len(data[i].unique()) > 2:
            return "One of those fields has more than 2 possible values, please rectify and then retry"
        else:
            continue
    num1 = int(input(f"how many of the following fields are neither the predictor or predictants? {columns}"))
    to_drop = []
    for i in range(num1):
        to_drop.append(0)
    for i in range(num1):
        to_drop[i] = input(f"one at a time, select those columns from the following: {columns} ")
    reduced = data.drop(to_drop, axis = 1)
    reduced_columns = []
    for i in (reduced):
        reduced_columns.append(i)
    nulls = reduced.isnull().sum()
    large_null = []
    for i in reduced:
        if (nulls[i] > 0) & (nulls[i] <= 30):
            reduced.dropna(subset = [i], inplace=True)
            print (f"The few rows with null values for the field '{i}' have been removed")
        elif (nulls[i] > 30) & (nulls[i] < (np.mean(reduced.count()))):
            print (f"the field {i}, has too many null values, these need to first be filled in using averages")
            large_null += [i]
            if len(large_null) > 1:
                return "sorry, too many fields have too many null fields, please rectify  and try again"
            else:
                continue
        else:
            continue
    if len(large_null) > 0:
        grouping_field = input(f"choose a field to group {large_null[0]} by from the following: {reduced_columns}")
    grouping = reduced.groupby(reduced[(grouping_field)])[large_null]
    means = (grouping.mean())
    reduced[large_null] = reduced[large_null].apply(avg_approx, axis = 1, args = (reduced, grouping_field, large_null[0], means))
    print (reduced.isnull().sum())



multLogReg()




avg_approx(reduced, grouping_field, large_null[0], means)
