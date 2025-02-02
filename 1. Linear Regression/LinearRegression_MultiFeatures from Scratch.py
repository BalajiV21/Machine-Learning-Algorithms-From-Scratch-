import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("multiple_linear_regression_dataset.csv")

data.info()
X = data.drop(columns = ['income'])
y = data['income']

from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_scaled = standardscaler.fit_transform(X)
X_scaled

m,n = X_scaled.shape 
w = np.zeros((n,1))
b = 0 
n_iteration = 10000
learning_rate=0.002
y=y.values.reshape(-1,1)

for i in range(n_iteration):
    y_pred = np.dot(X_scaled,w)+b

    dw = (1/m)*np.dot(X_scaled.T, (y_pred-y))
    db = (1/m)*np.sum(y_pred-y)

    w -= learning_rate*dw
    b -= learning_rate*db

    if i%1000 ==0:
        loss = (1/2*m)*np.sum((y_pred -y)**2)
        print(f"iteration:{i}, Loss:{loss:.4f}, w:{w}, b:{b}")


prediction_value1 = np.array([[25,0]])
prediction_value1 = standardscaler.transform(prediction_value1)

pricevalue = np.dot(prediction_value1,w) + b

print(f"Income for age {prediction_value1[0][0]},experience {prediction_value1[0][1]} : ${pricevalue[0][0]:.2f}") 