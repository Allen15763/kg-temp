import numpy as np
import pandas as pd

from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350

    params_nn = {
        'layers': 3,
        # 因為是範例程式，因此只設定 5 個 epoch 讓程式早一點結束，實際上可以調大一點
        'nb_epoch': 5,  # 1000
        'patience': 10,
        'dropout': 0.5,
        'units': 512,
    }

    # 設定特徵的名字
    features = [f'feat_{i}' for i in range(1, 94)]

    # # 訓練 xgboost 模型，並做出預測
    # runner = Runner('xgb1', ModelXGB, features, params_xgb)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('xgb1')
    #
    # # 訓練類神經網路，並做出預測
    # runner = Runner('nn1', ModelNN, features, params_nn)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('nn1')

    '''
    # (参考）使用所有資料來訓練 xgboost 模型，並做出預測
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_predict_all()
    Submission.create_submission('xgb1-train-all')
