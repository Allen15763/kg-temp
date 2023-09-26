# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用 Pandas 的 DataFrame 存資料

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# 定義模型的 Method
import xgboost as xgb


class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# hold-out 法
# -----------------------------------
# 使用 train_test_split() 劃分訓練、驗證資料
# 訓練資料佔 75%，驗證資料佔 25%
# 劃分之前先隨機打亂資料

from sklearn.model_selection import train_test_split

tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# -----------------------------------
# 訓練、驗證模型
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# 使用 train_test_split() 劃分訓練、驗證資料
# 訓練資料佔 75%，驗證資料佔 25%
# 劃分之前先隨機打亂資料
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# 透過 Model 類別來建立模型
# 使用 fit() 進行訓練
# 使用 predict() 預測驗證資料
# 計算 logloss 分數
model = Model()
model.fit(tr_x, tr_y, va_x, va_y)
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(score)

# -----------------------------------
# 使用 Kfold() 進行 hold-out 法
# -----------------------------------
from sklearn.model_selection import KFold

# 以 Kfold() 來進行 hold-out 法的資料劃分
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 使用 Kfold() 來進行交叉驗證中的資料劃分
# -----------------------------------

from sklearn.model_selection import KFold

# 使用 Kfold() 來進行交叉驗證的資料劃分 (進行 4 Fold 劃分)
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 交叉驗證
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Modelクラスを定義しているものとする
# Modelクラスは、fitで学習し、predictで予測値の確率を出力する

scores = []

# 使用 Kfold() 以交叉驗證來進行劃分資料
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 透過 Model 類別建立模型
    # 訓練模型
    # 預測驗證資料
    # 計算 logloss 分數
    # 儲存此 fold 分數
    model = Model()
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 各 fold 的平均 logloss 分數
print(np.mean(scores))

# -----------------------------------
# Stratified K-Fold
# -----------------------------------
from sklearn.model_selection import StratifiedKFold

# 使用 StratifiedKFold 方法來進行分層抽樣的劃分
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# GroupKFold
# -----------------------------------
# 假設每 4 筆數據有一個相同的 user_id
train_x['user_id'] = np.arange(0, len(train_x)) // 4
# -----------------------------------

from sklearn.model_selection import KFold, GroupKFold

# 以 user_id 欄的顧客 ID 為單位進行劃分
user_id = train_x['user_id']
unique_user_ids = user_id.unique()

# 使用 Kfold (以 user_id 為單位進行劃分)
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_group_idx, va_group_idx in kf.split(unique_user_ids):
    # 將 user_id 劃分為 train/valid（使用於訓練的資料、驗證資料）
    tr_groups, va_groups = unique_user_ids[tr_group_idx], unique_user_ids[va_group_idx]

    # 根據各筆資料的 user_id 是屬於 train 或 valid 來進行劃分
    is_tr = user_id.isin(tr_groups)
    is_va = user_id.isin(va_groups)
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

#（參考）GroupKFold 類別中不能設定 shuffle、亂數種子因此較難使用
kf = GroupKFold(n_splits=4)
for tr_idx, va_idx in kf.split(train_x, train_y, user_id):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# leave-one-out
# -----------------------------------
# 假設只有 100 筆數據
train_x = train_x.iloc[:100, :].copy()
# -----------------------------------
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for tr_idx, va_idx in loo.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
