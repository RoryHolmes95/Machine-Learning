import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sb
import sklearn as sk

from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

address = 'titanic-training-data.csv'
file = pd.read_csv(address)
file.head()
file.describe()
file.info()

#check only two entries for binary fields
sb.countplot(x = 'Survived', data = file)
sb.countplot(x = 'Sex', data = file)

#check for null values
file.isnull().sum()

#remove fields that are not important (neither predictant nor predictors)
titanic_data = file.drop(['PassengerId', 'Ticket', 'Name'], axis = 1)
print (titanic_data)

#fill in missing values with averages
sb.boxplot(y = 'Age', x = 'SibSp', data = titanic_data)

def ageapprox(x):
    Age = x[0]
    SibSp = x[1]
    if pd.isnull(Age):
        if SibSp == 0:
            return 29
        elif SibSp == 1:
            return 30
        elif SibSp == 2:
            return 21
        elif SibSp == 3:
            return 10
        elif SibSp == 4:
            return 8
        elif SibSp == 5:
            return 13
        elif SibSp == 8:
            return 27
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age', 'SibSp']].apply(ageapprox, axis = 1)

titanic_data.isnull().sum()

#too many nulls for cabin to be worth it, get rid
titanic = titanic_data.drop(['Cabin'], axis = 1)

titanic.isnull().sum()

#drop rows for the two null remaining
titanic.dropna(inplace = True)

#convert categorical data to binary indicators
label_encoder = preprocessing.LabelEncoder()
gender = titanic['Sex']
gender_encoded = label_encoder.fit_transform(gender)
gender_encoded[0:10]
gender_DF = pd.DataFrame(gender_encoded, columns = ['Is_Male?'])
gender_DF.head()

embark = titanic['Embarked']
embark_encoded = label_encoder.fit_transform(embark)
one_hot = preprocessing.OneHotEncoder(categories = 'auto')
embark1hot = one_hot.fit_transform(embark_encoded.reshape(-1,1))
embark1hot_array = embark1hot.toarray()
embark_DF = pd.DataFrame(embark1hot_array, columns = ['C', 'Q', 'S'])
embark_DF.head()

titanic_mod = titanic.drop(['Sex', 'Embarked'], axis = 1)
titanic_DF = pd.concat([titanic_mod, gender_DF, embark_DF], axis = 1)
titanic_DF.head()
titanic_DF.isnull().sum()
titanic_DF.dropna(inplace=True)


#check all variables are independent
sb.heatmap(titanic_DF.corr())
titanic_DF.drop(['Pclass'], axis = 1)

#check dataset is large enough (need 50 rows per predictant)
titanic_DF.info()
titanic_DF.isnull().sum()

#split into testing and training set
predictant_train, predictant_test, predictor_train, predictor_test = model_selection.train_test_split(titanic_DF.drop(['Survived'], axis = 1), titanic_DF['Survived'], test_size = 0.2, random_state = 200)
print (predictor_train.shape)
print (predictant_train.shape)
predictant_train[0:5]

#fit to logistic regression line
LogReg = linear_model.LogisticRegression(solver = 'liblinear')
LogReg.fit(predictant_train, predictor_train)
predictant_train.isnull().sum()

#set up predictor
predictor = LogReg.predict(predictant_test)
print (predictor[0:10])

#check for precision
print (metrics.classification_report(predictor_test, predictor))

#test prediction
titanic_DF[750:751]
random_passenger = np.array([2.0, 6.0, 0.0, 1.0, 12.475, 0.0, 0.0, 0.0, 1.0]).reshape(1,-1)
print (LogReg.predict(random_passenger))
print (LogReg.predict_proba(random_passenger))
