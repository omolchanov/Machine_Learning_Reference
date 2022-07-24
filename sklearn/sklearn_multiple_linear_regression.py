from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# print cols side by side as it's supposed to be
pd.set_option("expand_frame_repr", False)

# Loading the dataset
df = pd.read_csv('../assets/petrol_consumption.csv')
print(df.describe().round(2).T)

# Dataset visualization. The independent variables and dependent
variables = df.columns[:-1]

for var in variables:
    plt.figure()
    sns.regplot(x=var, y='Petrol_Consumption', data=df).set(title=f'{var} and Petrol Consumption')
    plt.show()

# Calculating and visualizing correlation between variables
correlations = df.corr()
sns.heatmap(correlations, annot=True).set(title='Heatmap of Consumption Data - Pearson Correlations')
plt.show()

# Preparing the data
X = df[['Average_income', 'Paved_Highways', 'Population_Driver_licence(%)', 'Petrol_tax']]
y = df['Petrol_Consumption']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Training the model with the training set (x_train, y_train)
model = LinearRegression()
model.fit(x_train, y_train)

# Interception and coefficients
print('Line intercept: %s; Coefficients: %s \n' % (model.intercept_, model.coef_))

df_coef = pd.DataFrame(data=model.coef_, index=X.columns, columns=['Coef value'])
print(df_coef)

# Model evaluation
y_pred = model.predict(x_test)
print(pd.DataFrame({'Actual results': y_test, 'Predicted results': y_pred, 'Difference': y_test - y_pred}))

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

score_test = model.score(x_test, y_test)
score_train = model.score(x_train, y_train)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

print(f'Test score: {score_test * 100:.2f}%')
print(f'Train score: {score_train * 100:.2f}% \n')

# Making prediction
x_pred = pd.DataFrame({
    'Average_income': 3000,
    'Paved_Highways': 1500,
    'Population_Driver_licence(%)': 0.7,
    'Petrol_tax': 5}, index=[0])

y_pred = model.predict(x_pred)
print('Multiply Regression prediction \n %s \n Result: %s' % (x_pred, round(y_pred[0])))
