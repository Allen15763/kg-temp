import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union

from util import Logger, Util

logger = Logger()

"""
主要進行含有交叉驗證的訓練/評價/預測。
調用Model class、特徵清單、超參數
若使用各fold的模型平均來進行預測十，不使用run_train_all、run_predict_all方法。
"""

class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
        """建構子

        :param run_name: run 的名稱
        :param model_cls: 模型的類型
        :param features: 特徵清單
        :param params: 超參數
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """針對交叉驗證的某一個 fold 來進行模型訓練跟驗證

        除了跟其他 method 一起使用，也可以單獨使用這個 method 來調整參數等等

        :param i_fold: 第幾個 fold
        :return: 訓練好的模型、驗證資料的索引、驗證資料的預測結果、評價分數
        """
        # 讀取資料
        validation = i_fold != 'all'
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # 將資料分割成訓練資料跟驗證資料
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 訓練模型
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # 預測驗證資料，並算出評價分數
            va_pred = model.predict(va_x)
            score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)

            # 傳回訓練好的模型、驗證資料的索引、驗證資料的預測結果、評價分數
            return model, va_idx, va_pred, score
        else:
            # 使用所有資料來訓練模型
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # 傳回訓練好的模型
            return model, None, None, None

    def run_train_cv(self) -> None:
        """使用交叉驗證來訓練、驗證模型

        除了訓練、驗證模型之外，也會每一個 fold 的模型以及分數
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 進行交叉驗證 
        for i_fold in range(self.n_fold):
            # 訓練模型
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # 儲存模型
            model.save_model()

            # 蒐集預測結果以及分數
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 合併各個 fold 的結果
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 儲存預測值
        Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        # 儲存分數
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """使用交叉驗證過程中訓練的模型來對測試資料進行預測

        必須先執行 run_train_cv
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # 使用每一個 fold 訓練而得的 model 來對測試資料進行預測
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 將預測值平均
        pred_avg = np.mean(preds, axis=0)

        # 儲存預測值
        Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """用所有資料訓練模型，並儲存模型"""
        logger.info(f'{self.run_name} - start training all')

        # 用所有資料訓練模型
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """用所有資料訓練模型後，使用此模型對測試資料進行預測

        必須先執行 run_train_all
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 使用訓練後的模型對測試資料進行預測
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 儲存預測結果
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """對每一個交叉驗證的 fold 建立一個模型

        :param i_fold: 第幾個 fold
        :return: 模型 instance
        """
        # 根據 run 的名字、fold 數、模型類型產生名字
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """讀取訓練資料的特徵

        :return: 傳回訓練資料的特徵
        """
        # 讀取訓練資料的特徵
        # 如果想要讀取某些欄位，則需要修改這個 method
        # 讀取資料的效率很低，因此要根據資料讀取適當的量
        return pd.read_csv('../input/train.csv')[self.features[:2]] # 這個資料特徵只有2維+1維標籤

    def load_y_train(self) -> pd.Series:
        """讀取訓練資料的標籤

        :return: 傳回訓練資料的標籤
        """
        # 讀取訓練資料的標籤
        train_y = pd.read_csv('../input/train.csv')['target']
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """讀取測試資料的特徵

        :return: 傳回測試資料的特徵
        """
        return pd.read_csv('../input/test.csv')[self.features[:2]] # 這個資料只有2維

    def load_index_fold(self, i_fold: int) -> np.array:
        """傳回每個交叉驗證的 fold 裡資料的索引

        :param i_fold: 第幾個 fold
        :return: 每個 fold 裡資料的索引
        """
        # 傳回訓練資料、驗證資料的索引
        # 這裡的亂數種子是固定值，還有一個方法是每次的亂數種子存在一個文件
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]
