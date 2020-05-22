import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn import linear_model
scale = preprocessing.StandardScaler()

def multreg():
    """function that takes a dataset, chooses the predictant and predictors,
    and then returns the regression score"""
    file = input("copy and paste your data set csv here")
    dataset = pd.read_csv(file)
    num_of_fields = len(dataset.columns)
    fields = []
    for i in range(num_of_fields):
        fields.append(dataset.columns[i])
    predictant = input(f"Choose your predictant: {fields} ")
    predictor1 = input(f"Choose your first predictor: {fields} ")
    predictor2 = input(f"Choose your second predictor: {fields} ")
    predictors = [predictor1, predictor2]
    predictor_values = scale.fit_transform((dataset[predictors].values))
    predictant_values = dataset[predictant].values
    test = dataset.isnull().any()
    if test[predictors[0]] == True | test[predictors[1]] == True:
        return "You have missing values in your predictors, please rectify"
    else:
        LinReg = linear_model.LinearRegression()
        LinReg.fit(predictor_values, predictant_values)
        score = LinReg.score(predictor_values, predictant_values)
        print (f"The performance score for these predictants is {score:3.3}")
        pred1 = input(f"choose a value for the {predictors[0]}: ")
        pred2 = input(f"choose a value for the {predictors[1]}: ")
        scaleback = scale.transform([[pred1, pred2]])
        print (f"Based on your predictions, the {predictant} would be approx. {(LinReg.predict([scaleback[0]])[0]):0<10.8}")

multreg()
