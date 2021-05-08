import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
print('\n')
print(diabetes.keys())
print('\n')
# print(diabetes.DESCR)
# print(diabetes.data)
# print(diabetes.feature_names)

diabetes_X = np.array([[17], [18], [19]])

diabetes_X_train = diabetes_X
diabetes_X_test = np.array([[20]])

diabetes_Y_train = np.array([172, 174, 174.5])
diabetes_Y_test = np.array([175.25])

# x=feature y=lebel

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)
print('height:', diabetes_Y_predict)
print("Mean Square Error: ", mean_squared_error(
    diabetes_Y_test, diabetes_Y_predict))

print('Weights: ', model.coef_)
print('Intercepts: ', model.intercept_)

plt.scatter(diabetes_X_train, diabetes_Y_train)
plt.plot(diabetes_X_test, diabetes_Y_predict)
plt.show()
