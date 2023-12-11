import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from rerunner import ReRunner
from util import Submission

if __name__ == '__main__':

    

    def load_x_train() -> pd.DataFrame:
        return pd.read_csv('../input/train.csv').iloc[:, :-1]

    def load_y_train() -> pd.Series:
        train_y = pd.read_csv('../input/train.csv')['target']
        # train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test() -> pd.DataFrame:
        return pd.read_csv('../input/test.csv')
    tr_x, tr_y, va_x = load_x_train(), load_y_train(), load_x_test()

    features = [f'feat_{i}' for i in range(1, 94)] # not important on rerunner
    # print(features, tr_y.unique().__len__(), sep='\n')


    le = LabelEncoder()
    tr_y = le.fit_transform(tr_y)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    tr_y = pd.Series(tr_y)

    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': tr_y.unique().__len__(),
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10,
        'early_stopping_rounds': 10,
        'validation_classes': tr_y.unique(),
        'labelEncoder': le,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 25

    params_nn = {
        'layers': 3,
        'nb_epoch': 5,  # 1000
        'patience': 10,
        'dropout': 0.5,
        'units': 512,
    }

    # runner = ReRunner('xgb_gl', ModelXGB, features, params_xgb, tr_x, tr_y)
    # runner.run_train_cv()
    # runner = ReRunner('xgb_gl', ModelXGB, features, params_xgb, va_x, 'na')
    # runner.run_predict_cv()
    # Submission.create_submission('xgb_gl')


    runner = ReRunner('xgb_gl-train-all', ModelXGB, features, params_xgb, tr_x, tr_y)
    runner.run_train_all()
    runner = ReRunner('xgb_gl-train-all', ModelXGB, features, params_xgb, va_x, 'na')
    runner.run_predict_all()
    Submission.create_submission('xgb_gl-train-all')
    
    # runner = ReRunner('xgb2', ModelXGB, features, params_xgb, tr_x, tr_y)
    # runner.run_train_cv()
    # # runner = ReRunner('xgb2', ModelXGB, features, params_xgb, va_x, 'na')
    # runner.run_predict_cv()
    # Submission.create_submission('xgb2')
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
    # runner = ReRunner('xgb1-train-all', ModelXGB, features, params_xgb_all, tr_x, tr_y)
    # runner.run_train_all()
    # runner = ReRunner('xgb1-train-all', ModelXGB, features, params_xgb_all, va_x, 'na')
    # runner.run_predict_all()
    # Submission.create_submission('xgb1-train-all')
