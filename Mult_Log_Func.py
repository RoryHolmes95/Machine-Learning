import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sb
import sklearn as sk

from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

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
    print (lst)
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
    print (to_drop)
    reduced = data.drop(to_drop, axis = 1)
    nulls = reduced.isnull().sum()
    for i in reduced:
        print (i)
        if (nulls[i] > 0) & (nulls[i] <= 30):
            reduced.dropna(subset = [i], inplace=True)
            print ("The few rows with null values for this field have been removed")
        elif (nulls[i] > 30):
            print (f"Sorry, too many null values in the field {i}, please rectify and then try again")
            reduced = reduced.drop([i], axis = 1)    
        else:
            continue
    print (reduced.info())

multLogReg()
