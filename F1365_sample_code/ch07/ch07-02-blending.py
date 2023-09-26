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

# 類神經網路需要的資料
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# 執行使用 hold-out 資料的預測值來集成
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_index = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_index]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_index]
tr_x_nn, va_x_nn = train_x_nn.iloc[tr_idx], train_x_nn.iloc[va_index]

# 假設 models.py 已經定義好參與集成的模型
# 訓練好每個模型之後，輸出預測機率
from models import Model1Xgb, Model1NN, Model2Linear

# 第 1 層模型
# 用訓練資料來訓練模型，接著對 hold-out 資料以及測試資料做預測
model_1a = Model1Xgb()
model_1a.fit(tr_x, tr_y, va_x, va_y)
va_pred_1a = model_1a.predict(va_x)
test_pred_1a = model_1a.predict(test_x)

model_1b = Model1NN()
model_1b.fit(tr_x_nn, tr_y, va_x_nn, va_y)
va_pred_1b = model_1b.predict(va_x_nn)
test_pred_1b = model_1b.predict(test_x_nn)

# 評價 hold-out 資料的預測精準度
print(f'logloss: {log_loss(va_y, va_pred_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(va_y, va_pred_1b, eps=1e-7):.4f}')

# 將 hold-out 資料以及測試資料的預測值視為第 2 層模型的訓練資料
va_x_2 = pd.DataFrame({'pred_1a': va_pred_1a, 'pred_1b': va_pred_1b})
test_x_2 = pd.DataFrame({'pred_1a': test_pred_1a, 'pred_1b': test_pred_1b})

# 第 2 層模型
# 我們把所有 hold-out 資料的預測值都視為第 2 層模型的訓練資料
# 因此沒有驗證資料可以用來評價模型
# 讀者可以考慮對 hold-out 資料做交叉驗證，便可以評價模型
model2 = Model2Linear()
model2.fit(va_x_2, va_y, None, None)
pred_test_2 = model2.predict(test_x_2)
