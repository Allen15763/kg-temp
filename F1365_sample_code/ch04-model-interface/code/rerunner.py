from typing import Callable, List, Optional, Tuple, Union
from sklearn.metrics import log_loss, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from model import Model
from runner import Runner
from util import Logger, Util
from sklearn.preprocessing import OneHotEncoder
import re

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
        # super().train_fold(i_fold)

        validation = i_fold != 'all'
        train_x = self.data_x
        train_y = self.data_y

        if validation:
            # 將資料分割成訓練資料跟驗證資料
            tr_idx, va_idx = self.load_index_fold(i_fold)
            print(tr_idx)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 訓練模型
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # 預測驗證資料，並算出評價分數
            va_pred = model.predict(va_x)
            # print(va_pred.shape, va_y.shape, va_pred.argmax(axis=1), sep='\n')
            # print(va_pred.argmax(axis=1)[:5], va_y.values[:5], sep='\n')
            # print(type(va_y.values), type(va_pred.argmax(axis=1)), sep='\n')
            # print(va_y.values.shape, va_pred.argmax(axis=1).shape, sep='\n')
            # print(log_loss(np.array([1,2,3]), np.array([1,2,4]), eps=1e-15, normalize=True))
            ol = OneHotEncoder(sparse=False)
            va_y = ol.fit_transform(va_y.values.reshape(-1, 1))
            # print(va_pred.shape, va_y.shape, sep='\n')
            # print(len(self.params['validation_classes']), self.params['validation_classes'], type(self.params['validation_classes']), sep='\n')
            # print(va_pred, sep='\n')

            # score = log_loss(va_y, va_pred, eps=1e-15, normalize=True, labels=self.params['validation_classes'])
            # score = classification_report(self.params['labelEncoder'].inverse_transform(va_y.argmax(axis=1))
            #                               , self.params['labelEncoder'].inverse_transform(va_pred.argmax(axis=1)))
            # score = float(re.search('macro avg\s{7}(.{4})\s{6}(.{4})\s{6}(.{4})', score).group()[-4:])
            
            score = f1_score(va_y.argmax(axis=1), va_pred.argmax(axis=1), average='micro')
            logger.info(f'f1 score:{score}')
            

            # 傳回訓練好的模型、驗證資料的索引、驗證資料的預測結果、評價分數
            return model, va_idx, va_pred, score
        else:
            # 使用所有資料來訓練模型
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # 傳回訓練好的模型
            return model, None, None, None
    
    def run_train_cv(self) -> None:
        # super().run_train_cv()
        
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score: {score}')

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
        # super().run_predict_cv()
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
        # super().run_train_all()
        logger.info(f'{self.run_name} - start training all')

        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.data_x

        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def load_index_fold(self, i_fold: int) -> np.array:
        # super().load_index_fold(i_fold)
        train_y = self.data_y
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]