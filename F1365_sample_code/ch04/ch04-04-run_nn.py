# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用 Pandas 的 DataFrame 存資料
# 資料經過 One-hot encoding

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 將具有標籤的資料分為訓練資料以及驗證資料
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx] # dataframe
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 關掉 Tensorflow 的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -----------------------------------
# 建立、訓練類神經網路模型
# -----------------------------------
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# 資料的縮放
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x) # dataframe to numpy.ndarray
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# 建立類神經網路模型
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],))) # 隱藏層
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu')) # 隱藏層
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # 輸出層

# 編譯模型
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 執行訓練
# 將驗證資料給模型，隨著訓練進度，觀察驗證分數的變化
batch_size = 128
epochs = 10
history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y))

# 確認驗證分數
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred, eps=1e-7)
print(f'logloss: {score:.4f}') # 0.2962

# 預測
pred = model.predict(test_x)

# -----------------------------------
# 提前中止
# -----------------------------------
from keras.callbacks import EarlyStopping

# 設定提前中止的監測為 round 20
# 透過設定 restore_best_weights 來使用最佳的 epoch 模型
epochs = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])
pred = model.predict(test_x)

va_pred = model.predict(va_x)
print(model.evaluate(va_x, va_y)) # evaluate這邊應該放測試集，但該範例沒有所以先放驗證集示範。
print(history.history.keys())
score = log_loss(va_y, va_pred, eps=1e-7)
print(f'logloss: {score:.4f}') # 0.2875
"""
這段程式碼用於建立、訓練和測試一個二元分類的神經網路模型，以下是它的主要步驟：
1.  資料準備：將讀取的訓練和測試資料進行 one-hot encoding 處理，並將訓練資料分為訓練集和驗證集。
2.  建立神經網路模型：使用 Keras 库建立一個類神經網路模型，包括三個全連接層，其中前兩個層都包含有 256 個神經元並使用 ReLU 激活函數，最後一層則包含一個神經元，使用 sigmoid 激活函數進行二元分類預測。
3.  編譯模型：使用二元交叉熵作為損失函數，adam 作為優化器，並以準確度作為評估指標進行編譯。
4.  訓練模型：使用 fit() 函数將訓練集和驗證集輸入模型進行訓練，同時通過 EarlyStopping 回調函數實現提前停止訓練的機制，以避免過度擬合。
5.  評估模型：使用 log-loss 作為評估指標，計算在驗證集上的預測分數。
6.  測試模型：使用訓練好的模型對測試集進行預測。
總之，這段程式碼是一個典型的神經網路二元分類模型的訓練過程，包括資料準備、模型建立、編譯、訓練、評估和測試等步驟。
"""