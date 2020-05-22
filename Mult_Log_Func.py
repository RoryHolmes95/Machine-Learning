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
    y = data[fill]
    length = len(data[fill])
    for each in row:
        if pd.isnull((y)[row.name]):
            for i in x:
                if i == data[group][row.name]:
                    return avgs[i]
        else:
            return (y[row.name])

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
        elif (nulls[i] > 30) & (nulls[i] <= ((np.mean(reduced.count()))/3)):
            print (f"The field '{i}', has too many null values, these need to first be filled in using averages")
            large_null += [i]
            if len(large_null) > 1:
                return "Sorry, too many fields have too many null fields, please rectify  and try again"
            else:
                continue
        elif nulls[i] > (np.mean(reduced.count())/3):
            reduced = reduced.drop(i, axis = 1)
            print (f"The field '{i}' had too many null values to be rectified and has been dropped")
        else:
            continue
    if len(large_null) > 0:
        grouping_field = input(f"choose a field to group {large_null[0]} by from the following: {reduced_columns}")
    grouping = reduced.groupby(reduced[(grouping_field)])[large_null[0]]
    means = (grouping.mean())
    reduced[large_null] = reduced[large_null].apply(avg_approx, axis = 1, args = (reduced, grouping_field, large_null[0], means))
    binaries = []
    num_of_binaries = int(input("How many of the binary fields are remaining? Please do not include the predictor"))
    label_encoder = preprocessing.LabelEncoder()
    if num_of_binaries > 0:
        for i in range(num_of_binaries):
            binaries += [0]
        for i in binaries:
            binaries[i] = input(f"One at a time, please list these fields: {reduced_columns}")
        for i in binaries:
            Var = reduced[i]
            encoded = label_encoder.fit_transform(Var)
            DF = pd.DataFrame(encoded, columns = [i])
            reduced = reduced.drop([i], axis = 1)
            reduced = pd.concat([reduced, DF], axis = 1)
    categoricals = []
    cat_data = int(input(f"How many of the remaining feilds contain categorical answers with more than 2 possible answers? {reduced_columns}"))
    reduced.dropna(inplace=True)
    if cat_data > 0:
        for i in range(cat_data):
            categoricals += [0]
        for i in categoricals:
            categoricals[i] = input(f"Once at a time, name those categories: {reduced_columns}")
        for i in categoricals:
            cat_var = reduced[i]
            cat_encoded = label_encoder.fit_transform(cat_var)
            onehot = preprocessing.OneHotEncoder(categories='auto')
            catvar1hot = onehot.fit_transform(cat_encoded.reshape(-1,1))
            make_array = catvar1hot.toarray()
            headers = []
            num_of_headers = int(input(f"how many different possible answers are there in the {i} field?"))
            for k in range(num_of_headers):
                headers += [0]
            for j in range(len(headers)):
                headers[j] = input("Once at a time, name those answers")
            cat_DF = pd.DataFrame(make_array, columns = headers)
            reduced = reduced.drop([i], axis = 1)
            reduced = pd.concat([reduced, cat_DF], axis = 1)
    reduced.dropna(inplace=True)
    print (reduced.info())
    print (reduced.isnull().sum())



multLogReg()
