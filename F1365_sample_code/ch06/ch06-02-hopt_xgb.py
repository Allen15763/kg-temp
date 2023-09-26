import numpy as np
from hyperopt import hp

# -----------------------------------
# xgboost 參數空間
# -----------------------------------

# 基準值
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.0,
    'alpha': 0.0,
    'lambda': 1.0,
    'min_child_weight': 1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 71,
}

# 參數的搜索範圍
param_space = {
    'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),
    'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
    # 如果程式執行速度夠快，且時間充裕，請調整 Alpha 和 Lambda
    # 'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
    # 'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
}
