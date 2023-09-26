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

# 假設資料具有時間序列，且變數 Period 代表每筆資料的時間點
train_x['period'] = np.arange(0, len(train_x)) // (len(train_x) // 4)
train_x['period'] = np.clip(train_x['period'], 0, 3)
test_x['period'] = 4

# -----------------------------------
# 時間序列資料的 hold-out 法
# -----------------------------------
# 以變數 period 作為劃分的基準 (從 0 到 3 為訓練資料，4 為測試資料)
# 從訓練資料中將變數 period 為 3 的資料做為驗證資料，0 到 2 的資料做為訓練用的資料
is_tr = train_x['period'] < 3
is_va = train_x['period'] == 3
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]

# -----------------------------------
# 時間序列資料的交叉驗證 (依時序進行驗證)
# -----------------------------------
# 以變數 period 為基準進行劃分（從 0 到 3 為訓練資料，4 為測試資料）
# 將變數 period 為 1, 2, 3 的資料作為驗證資料，比驗證資料更早以前的資料則作為訓練資料

va_period_list = [1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] < va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# (參考) 也可使用 TimeSeriesSplit，但只能依據資料的排序劃分，使用上較為不便
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=4)
for tr_idx, va_idx in tss.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 時間序列資料中的交叉驗證 ( 不管時序直接劃分資料的方法)
# -----------------------------------
# 以變數 period 為基準劃分資料 (0 到 3 為訓練資料，4 為測試資料)
# 將變數 period 為 0, 1, 2, 3 的資料作為驗證資料，其他的訓練資料則用於訓練模型

va_period_list = [0, 1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] != va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
