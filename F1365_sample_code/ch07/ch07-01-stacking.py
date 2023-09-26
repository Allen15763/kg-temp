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
# 執行堆疊
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 在 models.py 中已定義 Model1Xgb, Model1NN, Model2Linear 模型
# 各模型以 fit 進行訓練、以 predic t 輸出預測值機率

from models import Model1Xgb, Model1NN, Model2Linear


# 產生訓練資料跟測試資料預測值的函式 (在未知標籤的情況下產生訓練資料的預測值)
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # 在交叉驗證中進行訓練/預測，並保存預測值及索引
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # 將驗證資料的預測值整合起來，並依序排列
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # 取測試資料的預測值平均
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test


# 第 1 層的模型
# pred_train_1a, pred_train_1b 為訓練資料在交叉驗證時得到的預測值
# pred_test_1a 和 pred_test_1b 是測試資料的預測值
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train_x, train_y, test_x)

model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_x_nn, train_y, test_x_nn)

# 對第 1 層模型的評價
print(f'logloss: {log_loss(train_y, pred_train_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(train_y, pred_train_1b, eps=1e-7):.4f}')

# 將預測值作為特徵並建立 dataframe
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

# 第 2 層模型
# pred_train_2 為第 2 層模型的訓練資料預測值，由交叉驗證後獲得
# pred_test_2 為第 2 層模型的測試資料預測值
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, train_y, test_x_2)
print(f'logloss: {log_loss(train_y, pred_train_2, eps=1e-7):.4f}')
