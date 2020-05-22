import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
scale = preprocessing.StandardScaler()

#assign filename a variable
file_address = 'cars.csv'

#read in file/dataset
cars = pd.read_csv(file_address)

cars.head()
print (cars.columns[0:5])
#look for correlation
sb.pairplot(cars)
cars.corr()

#assign predictors a variable and scale them
predictors = scale.fit_transform((cars[['Volume', 'Weight']].values))
print (predictors[0:5])

#assign predictant a variable and scale it
predictant = cars[['CO2']].values

#check for missing values
missing_values = (predictors == np.NaN)
x = predictors[missing_values == True]
len(x)
print (x)
test = cars.isnull().any()
if test['Weight'] == True | test['Volume'] == True:
    print ("tada")
else:
    print ("nah")

#assign linreg function a variable
linear_model1 = linear_model.LinearRegression()

#fit the predictors and predictant to the function
linear_model1.fit(predictors,predictant)

#see how accurate the model is
linear_model1.score(predictors,predictant)

#to predict, need to use scaled version
scaled = scale.transform([[1000, 790]])
print (scaled)

#input scaled versions into predictor version to get y-value
linear_model1.predict([scaled[0]])

#get gradient coefficients for each predictor
linear_model1.coef_

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
