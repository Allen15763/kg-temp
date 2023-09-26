import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 讀取資料、將訓練資料與測試資料整合在一個 Dataframe
df_train = pd.read_csv('train.csv')
df_train['is_train'] = True
df_test = pd.read_csv('test.csv')
df_test['target'] = 0
df_test['is_train'] = False

df = pd.concat([df_train, df_test], axis=0)

# 日期的預處理
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['yearmonth'] = df['year'] * 12 + df['month']
df = df.drop(['date'], axis=1)

# 特徴的種類
numerical_features = ['age', 'height', 'weight', 'amount', 'year', 'month', 'month', 'yearmonth'
                                                                                     'medical_info_a1',
                      'medical_info_a2', 'medical_info_a3', 'medical_info_b1']
binary_features = [f'medical_keyword_{i}' for i in range(10)]
categorical_features = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# 對類別變數進行 Label encoding
for c in categorical_features:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    print(f'{c} - {le.classes_}')

# 將 target 欄位移到最後面
df = df.reindex(columns=[c for c in df.columns if c != 'target'] + ['target'])

# 分別儲存訓練資料跟測試資料
train = df[df['is_train']].drop(['is_train'], axis=1).reset_index(drop=True)
test = df[~df['is_train']].drop(['is_train', 'target'], axis=1).reset_index(drop=True)
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)

# ----------------------
# 類神經網路、線性模型需要的預處理
# ----------------------
# 填補欠缺值
has_nan_features = ['medical_info_c1', 'medical_info_c2']
for c in has_nan_features:
    df[f'{c}_nan'] = df[c].isnull()
    df[c].fillna(df[c].mean(), inplace=True)

# 對類別變數執行 One-hot encoding
df_onehot = pd.DataFrame(None, index=df.index)
for c in df.columns:
    if c in categorical_features and df[c].nunique() > 2:
        dummies = pd.get_dummies(df[c], prefix=c)
        df_onehot = pd.concat([df_onehot, dummies], axis=1)
        print(f'one-hot encoded - {c}')
    else:
        df_onehot[c] = df[c]


# 將 target 欄位移到最後面
df_onehot = df_onehot.reindex(columns=[c for c in df_onehot.columns if c != 'target'] + ['target'])

# 分別儲存訓練資料跟測試資料
train_onehot = df_onehot[df_onehot['is_train']].drop(['is_train'], axis=1).reset_index(drop=True)
test_onehot = df_onehot[~df_onehot['is_train']].drop(['is_train', 'target'], axis=1).reset_index(drop=True)
train_onehot.to_csv('train_preprocessed_onehot.csv', index=False)
test_onehot.to_csv('test_preprocessed_onehot.csv', index=False)
