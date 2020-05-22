import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
from sklearn import linear_model

#choose address of dataset
file_address = 'C:/Users/rory/Downloads/Ex_Files_Python_Data_Science_EssT_Pt2/Ex_Files_Python_Data_Science_EssT_Pt2/Exercise Files/Data/enrollment_forecast.csv'

#read file into python using pandas
enroll_data = pd.read_csv(file_address)

enroll_data.head()

#check for correlating variables
sb.pairplot(enroll_data)
print (enroll_data.corr())

#Assign two data fields that you think are correlating to a variable (predictors)
enroll_data_x = enroll_data[['inc', 'hgrad']].values

#Assign the predictant to a variable
enroll_data_y = enroll_data[['roll']].values

#Scale where required and then assign both predictors and predictant to new variables
X, Y = preprocessing.scale(enroll_data_x), enroll_data_y

#check for missing values
missing_values = (X == np.NaN)
X[missing_values == True]

#apply sklearns multiple linear regression function and get a precision score
LinReg = linear_model.LinearRegression()
LinReg.fit(X,Y)
LinReg.score(X,Y)
print (LinReg.intercept_, LinReg.coef_)

#to predict predictant, must first unscale values or use scaled predictors below
LinReg.predict([[2000,11000]])
