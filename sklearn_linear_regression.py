import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# Training dataset
X = np.array([
    [2.5, 21],
    [5.1, 47],
    [3.2, 27],
    [8.5, 75],
    [3.5, 30],
])

df = pd.DataFrame(X)


# Analyzing the dataset: correlation and basic stat info
print(df.corr(), '\n')
print(df.describe(), '\n')


# Dividing the dataset
x = df[0].values.reshape(-1, 1)
y = df[1].values.reshape(-1, 1)


# Split the dataset (features, labels) onto training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

model = LinearRegression()


# Training the model with the training set (X_train, y_train)
model.fit(x_train, y_train)


# The equation that describes any straight line is: y = a ∗ x + b
# In this equation, y and x represent the dataset. b is where the line starts at the Y-axis, also called the Y-axis
# intercept and a defines if the line is going to be more towards the upper or
# lower part of the graph (the angle of the line), so it is called the slope of the line
# model._intercept_ is a, model.coef is b
print('Line intercept: %s; Line slope: %s \n' % (model.intercept_, model.coef_))


# Evaluating the model
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f} \n')

# Making a prediction
x_pred = np.array([[5.5], [10.2]])
y_pred = model.predict(x_pred)

print('Linear Regression predictions \n Features: %s \n Prediction Result: %s' %
       (str(x_pred).replace('\n', ''), (str(y_pred).replace('\n', ''))))


# Visualizing the dataset and predictions
x_data = np.concatenate((x, x_pred))
y_data = np.concatenate((y, y_pred))

plt.figure()
sns.regplot(x=x_data, y=y_data, data=df).set(title='Unvariative Linear Regression')
plt.show()
