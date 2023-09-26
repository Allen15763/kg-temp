import numpy as np
import pandas as pd

# ---------------------------------
# 隨機森林特徵的重要性
# ---------------------------------
train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
# ---------------------------------
from sklearn.ensemble import RandomForestClassifier

# 隨機森林
clf = RandomForestClassifier(n_estimators=10, random_state=71)
clf.fit(train_x, train_y)
fi = clf.feature_importances_

# 輸出重要性最高的特徵
idx = np.argsort(fi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], fi[idx][:5]
print('random forest importance')
print(top_cols, top_importances)

# ---------------------------------
# GBDT 特徵的重要性
# ---------------------------------
train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
# ---------------------------------
import xgboost as xgb

# xgboost
dtrain = xgb.DMatrix(train_x, label=train_y)
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50
model = xgb.train(params, dtrain, num_round)

# 輸出重要性較高的特徵
fscore = model.get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
print(fscore[:5])
