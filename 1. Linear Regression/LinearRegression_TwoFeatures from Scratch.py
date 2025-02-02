import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv("rounded_hours_student_scores.csv")
data.head()

X = data['Hours']
y = data['Scores']

w=0
b=0
n_iteration =2000
learning_rate = 0.01
m = len(y)


for i in range(n_iteration):
    y_pred = w*X+b
    dw = 1/m * np.sum((y_pred - y)*X)
    db = 1/m * np.sum(y_pred -y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i%50==0:
        loss = (1/(2*m)) * np.sum((y_pred - y)**2) 
        print(f"Iteration {i}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

new_X = np.array([[6]])  
predicted_y = w * new_X + b
print(f"Predicted score for 6 hours of study: {predicted_y[0][0]:.2f}")


plt.scatter(X,y, color="red")
plt.plot(X, w*X+b, color="blue", label = "regression Line")
plt.legend()
plt.show()