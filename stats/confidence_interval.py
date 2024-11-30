# Calculations of confidence interval
# https://www.statology.org/confidence-intervals-python/

import scipy.stats as st
import numpy as np

# Calculating Confidence Intervals Using the t Distribution
# https://stackoverflow.com/questions/17290661/interpretting-scipy-functions-meaning-and-usage-t-interval
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html
data = list(range(1, 5))
print('Dataframe: ', data)

conf_interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
print('Confidence interval: ', conf_interval)

# Confidence Intervals Using the Normal Distribution
np.random.seed(17)
data = np.random.randint(1, 150, 100)
print('\n', data)

conf_interval = st.norm.interval(confidence=0.95, loc=np.mean(data), scale=st.sem(data))
print('Confidence interval: ', conf_interval)
