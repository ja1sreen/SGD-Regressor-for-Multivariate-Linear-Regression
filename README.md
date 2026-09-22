# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Jaisree N
RegisterNumber: 212224060104
*/
```
```
# Code cell
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# Code cell
data = fetch_california_housing()

# Select first 3 features (for demonstration)
X = data.data[:, :3]   # shape (n_samples, 3)

# Create a multi-output target: [median_house_value, some_other_numeric_column]
# Here we use column index 6 (for demonstration) as the second output
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])
# Code cell
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)
# Code cell
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))
# Code cell
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)# Code cell
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   # predicted in scaled space
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         # back to original units
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    # ground-truth back to original

print("First 5 predictions (original units):")
print(Y_pred[:5])
# Code cell
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

# Per-output MSE (optional, helpful for debugging)
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)
# Code cell
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)


```



## Output:
```
First 5 predictions (original units):
[[ 1.12443839 35.73516826]
 [ 1.54510779 35.7563396 ]
 [ 2.3960357  35.38792669]
 [ 2.67214831 35.53488572]
 [ 2.11759153 35.66145465]]
Mean Squared Error (multi-output): 2.5786797117742917
MSE per output: [0.66307497 4.49428445]
Example 1
Inputs (raw): [ 1.6812     25.          4.19220056]
True outputs: [ 0.477 36.06 ]
Predicted   : [ 1.12443839 35.73516826]
----------------------------------------
Example 2
Inputs (raw): [ 2.5313     30.          5.03938356]
True outputs: [ 0.458 35.14 ]
Predicted   : [ 1.54510779 35.7563396 ]
----------------------------------------
Example 3
Inputs (raw): [ 3.4801     52.          3.97715472]
True outputs: [ 5.00001 37.8    ]
Predicted   : [ 2.3960357  35.38792669]
----------------------------------------
Example 4
Inputs (raw): [ 5.7376     17.          6.16363636]
True outputs: [ 2.186 34.28 ]
Predicted   : [ 2.67214831 35.53488572]
----------------------------------------
Example 5
Inputs (raw): [ 3.725      34.          5.49299065]
True outputs: [ 2.78 36.62]
Predicted   : [ 2.11759153 35.66145465]
----------------------------------------
```



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
