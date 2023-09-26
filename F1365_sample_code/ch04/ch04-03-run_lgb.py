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

# 將具有標籤的資料分為訓練資料以及驗證資料
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# from sklearn.model_selection import train_test_split
# # Split dataset into training and test sets
# tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

# -----------------------------------
# 使用 lightgbm
# -----------------------------------
import lightgbm as lgb
from sklearn.metrics import log_loss

# 將特徵和標籤轉換成 lightgbm 的資料結構
lgb_train = lgb.Dataset(tr_x, tr_y)
lgb_eval = lgb.Dataset(va_x, va_y)

# 設定超參數
params = {'objective': 'binary', 'seed': 71, 'verbose': 0, 'metrics': 'binary_logloss'}
num_round = 100

# 進行訓練
# 以超參數設定類別變數
# 將驗證資料餵給模型，一面進行訓練一面監控分數變化
categorical_features = ['product', 'medical_info_b2', 'medical_info_b3']
model = lgb.train(params, lgb_train, num_boost_round=num_round,
                  categorical_feature=categorical_features,
                  valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval])

# 計算驗證資料的 logloss 分數
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# 進行預測
pred = model.predict(test_x)

