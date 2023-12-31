from typing import Callable, List, Optional, Tuple, Union
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from model import Model
from runner import Runner
from util import Logger, Util

logger = Logger()


class ReRunner(Runner):
    def __init__(self, run_name: str
                 , model_cls: Callable[[str, dict], Model]
                 , features: List[str]
                 , params: dict
                 , x: pd.DataFrame
                 , y: pd.Series):
        super().__init__(run_name, model_cls, features, params)
        self.data_x = x
        self.data_y = y

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        super().train_fold(i_fold)

        validation = i_fold != 'all'
        train_x = self.data_x
        train_y = self.data_y

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
        super().run_train_cv()
        
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            model.save_model()

            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 合併各個 fold 的結果
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        logger.result_scores(self.run_name, scores)
    
    def run_predict_cv(self) -> None:
        super().run_predict_cv()
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.data_x

        preds = []

        # 使用每一個 fold 訓練而得的 model 來對測試資料進行預測
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        pred_avg = np.mean(preds, axis=0)

        Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        super().run_train_all()
        """用所有資料訓練模型，並儲存模型"""
        logger.info(f'{self.run_name} - start training all')

        # 用所有資料訓練模型
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def load_index_fold(self, i_fold: int) -> np.array:
        super().load_index_fold(i_fold)
        train_y = self.data_y
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]