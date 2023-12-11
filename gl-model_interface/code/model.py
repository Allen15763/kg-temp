import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """建構子

        :param run_fold_name: 包含 run 跟 fold 的名稱
        :param params: 超參數
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None) -> None:
        """訓練模型，並保存訓練後的模型

        :param tr_x: 訓練資料的特徵
        :param tr_y: 訓練資料的標籤
        :param va_x: 驗證資料的特徵
        :param va_y: 驗證資料的標籤
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """傳回預測值

        :param te_x: 驗證資料和測試資料的特徵
        :return: 預測值
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """儲存模型"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """讀取模型"""
        pass
