import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

df = pd.read_csv("Cancerstats.csv")

ohesex = pd.get_dummies(df['Sex'], prefix='Sex')

df = df.drop('Sex',axis = 1)
# Join the encoded df
df = df.join(ohesex)

df['log_bmi'] = np.log(1 + df['BMI'])

X = df[['Age','Sex_M','Sex_F','Tobaccofrequency','log_bmi','Alcoolfrequency']]

y = df['Cancer']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X)

mse = mean_squared_error(y,y_pred)

print(mse)
#from sklearn.metrics import mean_square