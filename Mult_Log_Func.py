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
    predictor = input("What is your chosen predictor?")
    cols = data.columns
    columns = []
    for i in (cols):
        columns.append(i)
    columns.remove(predictor)
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
    reduced_columns.remove(predictor)
    nulls = reduced.isnull().sum()
    large_null = []
    for i in reduced:
        if (nulls[i] > 0) & (nulls[i] <= 30):
            reduced.dropna(subset = [i], inplace=True)
            print (f"The few rows with null values for the field '{i}' have been removed")
        elif (nulls[i] > 30) & (nulls[i] <= ((np.mean(reduced.count()))/3)):
            print (f"The field '{i}', has too many null values, these will be filled in using user-defined averages")
            large_null += [i]
            if len(large_null) > 1:
                return "Sorry, too many fields have too many null values, please rectify and try again"
            else:
                continue
        elif nulls[i] > (np.mean(reduced.count())/3):
            reduced = reduced.drop(i, axis = 1)
            print (f"The field '{i}' had too many null values to be rectified and has been dropped")
        else:
            continue
    if len(large_null) > 0:
        grouping_field = input(f"'{large_null[0]}' has too many null values, choose a predictant to use in order to fill in the nulls with weighted averages: {reduced_columns}")
        grouping = reduced.groupby(reduced[(grouping_field)])[large_null[0]]
        means = (grouping.mean())
        reduced[large_null] = reduced[large_null].apply(avg_approx, axis = 1, args = (reduced, grouping_field, large_null[0], means))
    for name in reduced.corr():
        for each in reduced.corr()[name]:
            if ((each < -0.5) & (each != -1.0)) | ((each > 0.5) & (each != 1.0)):
                print (f"Unable to verify if two variables are independent, '{name}' has been dropped.")
                reduced = reduced.drop([name], axis=1)
    binaries = []
    num_of_binaries = int(input(f"How many of the following fields have binary answers? {reduced_columns}"))
    label_encoder = preprocessing.LabelEncoder()
    if num_of_binaries > 0:
        for i in range(num_of_binaries):
            binaries += [0]
        for i in binaries:
            binaries[i] = input(f"One at a time, please list these fields: {reduced_columns}")
        for i in binaries:
            if len(data[i].unique()) > 2:
                return "One of those fields has more than 2 possible values, please rectify and then retry"
        for i in binaries:
            Var = reduced[i]
            encoded = label_encoder.fit_transform(Var)
            DF = pd.DataFrame(encoded, columns = [i])
            reduced = reduced.drop([i], axis = 1)
            reduced = pd.concat([reduced, DF], axis = 1)
    if (np.mean(reduced.count())/len(reduced.columns)-1) > 50:
        print ("Enough predictants to proceed...")
    else:
        return "There are not enough fields for the number of predictants you have selected, please try again"
    categoricals = []
    red_cols = []
    for redcols in (reduced.columns):
        red_cols.append(redcols)
    red_cols.remove(predictor)
    cat_data = int(input(f"How many of the remaining fields contain categorical answers with more than 2 possible answers? {red_cols}"))
    reduced.dropna(inplace=True)
    if cat_data > 0:
        for i in range(cat_data):
            categoricals += [0]
        for i in range(cat_data):
            categoricals[i] = input(f"Once at a time, name those categories: {red_cols}")
        for i in categoricals:
            cat_var = reduced[i]
            cat_encoded = label_encoder.fit_transform(cat_var)
            onehot = preprocessing.OneHotEncoder(categories='auto')
            catvar1hot = onehot.fit_transform(cat_encoded.reshape(-1,1))
            make_array = catvar1hot.toarray()
            headers = []
            num_of_headers = len((reduced[i].unique()))
            for k in range(num_of_headers):
                headers += [0]
            for j in range(len(headers)):
                headers[j] = reduced[i].unique()[j]
            cat_DF = pd.DataFrame(make_array, columns = headers)
            reduced = reduced.drop([i], axis = 1)
            reduced = pd.concat([reduced, cat_DF], axis = 1)
            reduced.dropna(inplace=True)
    predictant_train, predictant_test, predictor_train, predictor_test = model_selection.train_test_split(reduced.drop([predictor], axis = 1), reduced[predictor], test_size = 0.2, random_state = 200)
    LogReg = linear_model.LogisticRegression(solver = 'liblinear')
    LogReg.fit(predictant_train, predictor_train)
    predicting = LogReg.predict(predictant_test)
    class_report = metrics.classification_report(predictor_test, predicting)
    cross = model_selection.cross_val_predict(LogReg, predictant_train, predictor_train, cv = 5)
    conf = metrics.confusion_matrix(predictor_train, cross)
    print (f"Out of {sum(sum(conf))} results, there were {conf[0][1]} false positives, and {conf[1][0]} false negatives")
    precision = metrics.precision_score(predictor_train, cross)
    test_passenger = []
    num_of_columns = len(reduced.drop([predictor], axis = 1).columns)
    for col in range(num_of_columns):
        test_passenger += [0]
    for num in range(len(test_passenger)):
        test_passenger[num] = int(input(f"One by one, fill in your predictions into the following predictants: {reduced.drop([predictor], axis = 1).columns}"))
        print (f"{reduced.drop([predictor],axis=1).columns[num]} : {test_passenger[num]}")
    test_passenger = np.array(test_passenger).reshape(1,-1)
    survived = (LogReg.predict(test_passenger))
    if survived[0] == 0:
        print (f"There is a {((LogReg.predict_proba(test_passenger)[0][0])*100):.3f}% chance that this passenger would have died.")
    else:
        print (f"There is a {((LogReg.predict_proba(test_passenger)[0][1])*100):.3f}% chance that this passenger would have survived.")
    return f"This prediction engine has a precision of {precision:.3f}"


multLogReg()
