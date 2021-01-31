import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('C:\\Users\\anish\\Documents\\DATA_SCIENCE_JUPYTER\\TECHNOCOLABS_INTERNSHIP\\newd.csv')
dataset.head()
dataset.isnull().sum()

X = dataset.iloc[:, 1:5]
y= dataset["skipped"]



#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state=0)
#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.4,0.3,0.5,0.6]]))
