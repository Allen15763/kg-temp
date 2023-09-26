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

import xgboost as xgb
from sklearn.metrics import accuracy_score

# 定義模型的 Method
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        """
        先把資料轉成更適合xgb使用的資料型態，
        之後的train method等於sk model的fit，會自動調用XGBClassifier。
        也可以初始化一個物件，xgb.XGBClassifier()，再調用裡面的fit。
        """
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

print(train_x.head(), train_y.head(), test_x.head(), sep='\n')
# -----------------------------------
# 模型的訓練及預測
# -----------------------------------
# 指定模型的超參數 (最大深度、學習率) for XGBClassifier
# params = {'param1': 10, 'param2': 100} # Parameters: { "param1", "param2", "silent" } might not be used.
params = {'max_depth': 10, 'learning_rate': 0.5} # learning_rate = eta

# 建立模型物件 (指定超參數)
model = Model(params)

# 透過訓練資料的特徵與標籤來訓練模型
model.fit(train_x, train_y)

# 對測試資料進行預測
pred = model.predict(test_x)


# -----------------------------------
# 訓練與驗證
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 將訓練資料分為 4 份，並將其中 1 份作為驗證資料
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 將資料分為訓練資料和驗證資料
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 定義模型
model = Model(params)

# 使用訓練資料訓練模型
model.fit(tr_x, tr_y)

# 以驗證資料進行預測並進行評價 (以 logloss 做為評價函數)
va_pred = model.predict(va_x) # 回傳的是機率
score = log_loss(va_y, va_pred)

predicted_labels = np.where(va_pred > 0.5, 1, 0) # > 0.5則判定True，來轉回標籤
print(f'logloss: {score:.4f}\nAccuracy: {accuracy_score(va_y, predicted_labels)}')

# -----------------------------------
# 交叉驗證
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 將訓練資料分為 4 等分，其中 1 份為驗證資料 
# 不斷輪替驗證資料，進行 4 次的訓練和評價 
scores = []
acc = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    predicted_labels = np.where(va_pred > 0.5, 1, 0)
    acc.append(accuracy_score(va_y, predicted_labels))
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 輸出交叉驗證的平均分數
print(f'logloss: {np.mean(scores):.4f}\nAccuracy: {np.mean(acc)}')
