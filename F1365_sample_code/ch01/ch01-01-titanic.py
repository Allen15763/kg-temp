import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# -----------------------------------
# 讀取訓練資料、測試資料
# -----------------------------------
# 讀取訓練資料、測試資料
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 分別取出訓練資料的特徵和標籤
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 由於測試資料只有特徵，維持原樣複製一份即可
test_x = test.copy()

# -----------------------------------
# 建立特徴
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 1. 去除 PassengerId
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 2. 去除 Name, Ticket, Cabin
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 3. 對 Sex、Embarked 進行 label encoding
for c in ['Sex', 'Embarked']:
    # 根據訓練資料決定如何轉換
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 訓練資料、測試資料的轉換
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# 建立模型
# -----------------------------------
from xgboost import XGBClassifier

# 建立模型及餵入訓練資料 (與標籤) 以進行學習
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 餵入測試資料以輸出預測值
pred = model.predict_proba(test_x)[:, 1]

# 將大於 0.5 的預測值轉為 1、小於等於 0.5 則轉成 0
pred_label = np.where(pred > 0.5, 1, 0)

# 建立提交用的檔案
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# 分數：0.7799 (可能跟書中的數值略有不同)
# -----------------------------------
# 驗證
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 用 List 保存各 fold 的 accuracy 與 logloss 分數
scores_accuracy = []
scores_logloss = []

# 進行交叉驗證
# 將資料分成四組，一組用於驗證，其餘三組用於訓練，並且輪替四次
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 將資料分為訓練資料和驗證資料 (標籤也是)
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 建立 xgboost 模型並餵入訓練資料與標籤進行學習
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # 對驗證資料進行預測，輸出預測值的準確率
    va_pred = model.predict_proba(va_x)[:, 1]

    # 計算驗證資料預測值的評價指標，用 logloss 及 accuracy 來算誤差
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 保存該 fold 的評價指標
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 輸出各 fold 評價指標的平均值
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148 (可能跟書中的數值略有不同)

# -----------------------------------
# 模型調整
# -----------------------------------
import itertools

# 準備用於調整的參數
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 產生超參數 max_depth 與 min_child_weight 的所有組合
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 用 List 保存各參數組合以及各組合的分數
params = []
scores = []

# 對各參數組合的模型進行n組交叉驗證，因為是分類模型所以使用訓練後模型概率vs驗證集標籤計算對數損失，以該母體平均作為該模型參數組合分
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # 進行交叉驗證
    # 將訓練資料分成4分，其中一個作為驗證資料，並不斷輪替交換
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 將資料分為訓練資料與驗證資料
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 建立 xgboost 模型並進行訓練
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 驗證資料的預測值與 logloss 評價指標
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 將各 fold 的評價指標進行平均
    score_mean = np.mean(score_folds)

    # 保存參數的組合以及其相對應的評價指標
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 找出將評價指標分數最佳的參數組合
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=3, min_child_weight=2.0 為最佳超參數組合；是(3, 2)


# -----------------------------------
# 建立邏輯斯迴歸模型所需特徵
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 訓練與測試資料
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 去除訓練、測試資料中的 PassengerId 欄位
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 去除訓練、測試資料中的 Name、Ticket、Cabin 欄位
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 建立 one-hot 編碼的物件與進行相關設置 (告訴它編碼的方式)
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# 對指定的欄位的數據進行 one-hot 編碼
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 對指定欄位的數據進行 one-hot 編碼，將結果存在一個新的 Dataframe
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# 去除原資料中，指定欄位的數據
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# 將 one-hot 編碼後的 Dataframe 與原資料合併
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 將欄位中的缺失值，用整個欄位的平均值取代
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 對 Fare 欄位的數據取對數
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# 集成
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# 建立 xgboost 模型並進行訓練與預測
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 建立邏輯斯迴歸模型並進行訓練與預測
# 必須放入與 xgboost 模型不同的特徵，因此另外建立了 train_x2,test_x2
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 取得加權平均後的預測值；把邏輯回歸分類器當成弱學習器給予較低權重分。
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)
