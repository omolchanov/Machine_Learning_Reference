from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import pandas as pd
import time

df = pd.read_csv('../assets/indian_flights.csv')

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regs = [
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100)
]

for i, r in enumerate(regs):
    start_time = time.time()

    r.fit(X_train, y_train)
    y_pred = r.predict(X_test)

    end_time = time.time()

    print('\n', r)
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))
    print('1 - MAPE: %.3f' % (1 - mean_absolute_percentage_error(y_test, y_pred)))
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print(f"Execution time: {end_time - start_time}s")
