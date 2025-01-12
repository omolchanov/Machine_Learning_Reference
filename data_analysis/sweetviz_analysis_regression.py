# https://www.geeksforgeeks.org/sweetviz-automated-exploratory-data-analysis-eda/
# https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
# https://library.soton.ac.uk/skewness-and-kurtosis

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate

import sweetviz as sv

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

'''
Dataframe's features and target value:
 
- MedInc        median income in block group
- HouseAge      median house age in block group
- AveRooms      average number of rooms per household
- AveBedrms     average number of bedrooms per household
- Population    block group population
- AveOccup      average number of household members
- Latitude      block group latitude
- Longitude     block group longitude

- MedHouseVal the median house value for California districts, expressed in hundreds of thousands of dollars (100K)
'''
df = fetch_california_housing(as_frame=True).frame

'''
A distinct number is a number that is not equal to another number in a set
Skewed data is data which is not normal. The two types of skewness are right-skew and left-skew. 

=====

A distribution with zero skew is one which is purely symmetrical, like Normal distributions. 
Another distribution with no skew is the uniform distribution. Positive skew indicates a distribution that is 
right-tailed, and a negative skew is indicative of a left-tailed distribution.

- Data with either a +1 or more, or -1 or less, is said to be highly positively/negatively skewed
- A more moderate positive or negative skewness lies between +0.5 and +1, or -0.5 and -1 respectively
- A very slight positive or negative skew lies between 0 and +1, or -0.5 and 0 respectively. In hypothesis testing, 
you can treat slightly skewed data as normal as it shows approximate symmetry
- Data with zero skew is said to be not skewed.

=====

Kurtosis is the heaviness in a histogram's tails.
The expected value of kurtosis is 3. This is observed in a symmetric distribution. A kurtosis greater than three will 
indicate Positive Kurtosis. In this case, the value of kurtosis will range from 1 to infinity. 
Further, a kurtosis less than three will mean a negative kurtosis. The range of values for a negative kurtosis is from 
-2 to infinity. The greater the value of kurtosis, the higher the peak. 
'''

# Identifying the most valuable features with Ridge coefs
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

alphas = np.logspace(1, 2, num=30)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

cv_results = cross_validate(model, X, y, return_estimator=True, cv=5, n_jobs=2)

score = cv_results['test_score']
print('R2 score: {:.3f} +- {:.3f}'.format(score.mean(), score.std()))

coefs = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=X.columns,
)

print(coefs)

coefs.plot.box(vert=False)
plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
plt.title("Coefficients of Ridge models\n via cross-validation")
plt.show()


# Building graph on location vs house value
sns.scatterplot(
    data=df,
    x='Longitude',
    y='Latitude',
    size='MedHouseVal',
    hue='MedHouseVal',
    palette='viridis',
)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
plt.title("Median house value depending of\n their spatial location")

plt.show()

# Generating report for the full dataframe
report = sv.analyze(df, target_feat='MedHouseVal')
report.show_html('ReportFull.html')

# Generating repost for train and test sets
train_df, test_df = train_test_split(df, train_size=0.75)
compare = sv.compare(source=train_df, compare=test_df, target_feat="MedHouseVal")
compare.show_html('CompareTrainTest.html')
