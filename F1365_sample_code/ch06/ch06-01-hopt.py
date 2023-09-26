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
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71} # baseline parameter
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
# 指定要進行搜索的參數空間
# -----------------------------------
# 以 hp.choice 從多個選項選出一個
# 以 hp.uniform 從已設定上/下限的均勻分布中選出一個數字。引數為上/下限。
# 以 hp.quniform 從已設定上/下限的均勻分布中，以一定間隔為單位從中選出一個數字。引數為下限/上限/間隔
# 以 hp.loguniform 從已設定上/下限的對數均勻分布中選出一個數字。引數為上/下限的對數。

from hyperopt import hp
# 僅範例，非下方實際套用之空間
space = {
    'activation': hp.choice('activation', ['prelu', 'relu']),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'units': hp.quniform('units', 32, 256, 32),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.01)),
}

# -----------------------------------
# 使用 hyperopt 探索參數
# -----------------------------------
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


def score(params):
    # 設定賦予參數時"最小化"的評價指標
    # 具體回傳模型以指定參數進行訓練、實行預測後得到的分數

    # 修改 max_depth 的形式為整數
    params['max_depth'] = int(params['max_depth'])

    # 建立 Model 物件
    # Model 的 fit() 進行訓練
    # 以 predict() 輸出預測值的機率
    model = Model(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    # score = accuracy_score(va_y, va_pred.round()) # 當改用準確率為評估指標
    print(f'params: {params}, logloss: {score:.4f}')

    # 記錄資訊，每組tuple裝參數組合跟目標函數評估分數
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}
    # return {'loss': -score, 'status': STATUS_OK} # 當改用準確率為評估指標 或 {'loss': score, 'status': STATUS_FALSE}


# 設定搜索的參數空間
space = {
    'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'gamma': hp.quniform('gamma', 0, 0.4, 0.1),
}

# 以 hyperopt 執行搜索參數
max_evals = 10
# len(trials) == max_evals，評估的結果會儲存在Trials()這個物件中
trials = Trials()
history = []
fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 從記錄的情報中輸出參數與分數
# 雖然從 trials 也可以取得資訊，但要取得參數並不那麼容易
history = sorted(history, key=lambda tpl: tpl[1]) # tuple(參數組合, 分數)，按分數排序
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')

# history = sorted(history, key=lambda tpl: tpl[1], reverse=True) # 當改用準確率為評估指標
