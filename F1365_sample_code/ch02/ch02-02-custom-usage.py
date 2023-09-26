# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用Pandas的DataFrame存資料

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 將具有標籤的資料分割成訓練資料跟驗證資料
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 在xgboost中使用自己定義的評價指標以及目標函數
# （参考）https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 透過 xgb.Dmatrix() 可將特徵與標籤資料轉換為適合 xgboost 模型的資料結構。這種資料結構可以提升記憶體的使用效率以及加快模型的訓練速度
# tr_x, tr_y為訓練資料的特徵與標籤、va_x, va_y為驗證資料的特徵與標籤
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)


# 自定義目標函數（此處其實是在實作 logloss、因此等同於 xgboost 的 ' binary:logistic'）
def logregobj(preds, dtrain):
    labels = dtrain.get_label()  # 取得實際值標籤
    preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid 函數
    grad = preds - labels  # 斜率
    hess = preds * (1.0 - preds)  # 二階導數值
    return grad, hess


# 自定義評價指標（此處為誤答率）
def evalerror(preds, dtrain):
    labels = dtrain.get_label()  # 取得實際值標籤
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)


# 設定超參數
params = {'silent': 1, 'random_state': 71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# 開始對模型進行訓練
bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

# 使用自定義目標函數的模型在進行預測時所輸出的預測值並非如同目標函數的輸出 (機率)，因此必須進行 Sigmoid 函數進行轉換
# 這與指定 binary:logistic 為目標函數不同，可參考下面的程式碼
pred_val = bst.predict(dvalid)
pred = 1.0 / (1.0 + np.exp(-pred_val))
logloss = log_loss(va_y, pred)
print(logloss)

# (參考) 使用一般訓練方法時使用，指定 binary:logistic 為目標函數
params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)
