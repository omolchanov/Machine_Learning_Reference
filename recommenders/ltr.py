"""
Learning-to-Rank is a machine learning approach to automatically order items based on relevance. It’s widely used in:

- Search engines (ranking search results)
- Recommendation systems (ranking products or movies)
- Ads placement

The goal: given a query or context, predict a ranking of items that maximizes relevance.

LTR algorithms are generally classified into three groups:

Pointwise

Treats ranking as a regression or classification problem for each item individually.
Example: Predict a relevance score for each document, then sort.
Pros: Simple to implement.
Cons: Doesn’t directly optimize the ordering.

Pairwise

Learns by comparing pairs of items.
Tries to predict which item in a pair is more relevant.
Pros: Optimizes relative order.
Common algorithms: RankSVM, RankNet.

Listwise

Considers the whole list of items at once.
Optimizes metrics like NDCG or MAP directly.
Pros: Most aligned with final ranking metrics.
Common algorithms: LambdaMART, ListNet, ListMLE.
"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Example data
data = pd.DataFrame({
    "query_id": [
        1, 1, 1, 1,  # query 1
        2, 2, 2,     # query 2
        3, 3, 3, 3,  # query 3
        4, 4, 4, 4,  # query 4
        5, 5, 5, 5, 5  # query 5
    ],
    "feature1": [
        0.1, 0.4, 0.35, 0.6,   # q1
        0.8, 0.7, 0.5,         # q2
        0.2, 0.3, 0.6, 0.9,    # q3
        0.15, 0.5, 0.55, 0.75, # q4
        0.1, 0.25, 0.45, 0.65, 0.85  # q5
    ],
    "feature2": [
        1, 0, 1, 0,   # q1
        0, 1, 0,      # q2
        1, 0, 1, 0,   # q3
        0, 1, 0, 1,   # q4
        1, 0, 1, 0, 1  # q5
    ],
    "relevance": [
        0, 1, 0, 1,   # q1
        1, 0, 0,      # q2
        0, 0, 1, 1,   # q3
        0, 1, 0, 1,   # q4
        0, 0, 1, 0, 1  # q5
    ]
})

X = data[["feature1", "feature2"]]
y = data["relevance"]
groups = data["query_id"]

# Split train/test by queries (not rows!)
gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
X_test,  y_test,  g_test  = X.iloc[test_idx],  y.iloc[test_idx],  groups.iloc[test_idx]

# Count group sizes for XGBoost
group_train = g_train.value_counts(sort=False).sort_index().to_list()
group_test = g_test.value_counts(sort=False).sort_index().to_list()

# DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)

dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_group(group_test)

# Parameters for pairwise ranking
params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "gamma": 1.0,
    "min_child_weight": 0.1,
    "max_depth": 3
}

# Train model
bst = xgb.train(params, dtrain, num_boost_round=10)

# Predict ranking scores
preds = bst.predict(dtest)
print("Predicted ranking scores:", preds)


df = pd.DataFrame({
    "query_id": g_test,
    "true_relevance": y_test,
    "score": preds
})

# Rank items per query
df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")
print(df.sort_values(["query_id", "rank"]))


