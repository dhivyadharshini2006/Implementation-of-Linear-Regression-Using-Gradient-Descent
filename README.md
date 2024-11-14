# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
 
1. Import necessary libraries: `numpy`, `pandas`, and `StandardScaler` from `sklearn`.
2. Define the `linear_regression` function to perform gradient descent and update `theta` iteratively.
3. Read the dataset from `50_Startups.csv` and print the first few rows to verify data.
4. Extract input features (`x`) and target variable (`y`) from the DataFrame.
5. Convert `x` to a float type for consistency.
6. Scale `x` and `y` using `StandardScaler` to normalize the data.
7. Train the linear regression model using the `linear_regression` function and obtain `theta`.
8. Create a new sample input (`new_data`) and scale it using the fitted `scaler`.
9. Make a prediction by computing the dot product of `theta` and the scaled input, including the intercept term.
10. Inverse-transform the predicted value to return it to the original scale.
11. Print the predicted value.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
Developed by: Dhivya Dharshini B
RegisterNumber: 212223240031
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range (num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv(r"C:\Users\admin\Downloads\50_Startups (1).csv",header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}") 
*/
```

## Output:

![image](https://github.com/user-attachments/assets/68ced7cc-5664-4290-b1b6-f94717baed62)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
