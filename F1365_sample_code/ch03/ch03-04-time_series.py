import numpy as np
import pandas as pd

# -----------------------------------
# 寬表格、長表格
# -----------------------------------

# 讀取寬表格
df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0)
# 將索引形態轉換成日期形態
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# 轉換成長表格
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# 還原成寬表格
df_wide = df_long.pivot(index=None, columns='id', values='value')

# -----------------------------------
# lag 變數
# -----------------------------------
# 設置寬格式數據
x = df_wide
print(f'x:\n{x}')
"""
              A     B     C
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
"""
# -----------------------------------
# x 為寬表格的 dataframe
# index 為日期等時間、列為使用者或店面等資料，值則為營業額等我們關注的變數

# 取得 1 個單位前的 lag 特徵；整個values往後移一列
x_lag1 = x.shift(1)
print(f'x_lag1:\n{x_lag1}')
"""
id              A       B       C
2016-07-01    NaN     NaN     NaN
2016-07-02  532.0  3314.0  1136.0
2016-07-03  798.0  2461.0  1188.0
"""
# 取得 7 個單位前的 lag 特徵
x_lag7 = x.shift(7)
print(f'x_lag7:\n{x_lag7}')
# -----------------------------------
# 計算前 1 ~ 3 單位期間的移動平均
x_avg3 = x.shift(1).rolling(window=3).mean()
print(f'x_avg3:\n{x_avg3}')
# -----------------------------------
# 計算前 1 單位到前 7 單位期間的最大值
x_max7 = x.shift(1).rolling(window=7).max()
print(f'x_max7:\n{x_max7}')
# -----------------------------------
# 將前 7 單位, 前 14 單位, 前 21 單位, 前 28 單位這些數值進行平均
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0
print(f'x_e7_avg:\n{x_e7_avg}')
# -----------------------------------
# 取得 1 單位後的值
x_lead1 = x.shift(-1)
print(f'x_lead1:\n{x_lead1}')
"""
id              A       B       C
2016-07-01  798.0  2461.0  1188.0
2016-07-02  823.0  3522.0  1711.0
2016-07-03  937.0  5451.0  1977.0
"""
# -----------------------------------
# 將資料與時間做連結的方法
# -----------------------------------
# 讀取資料
train_x = pd.read_csv('../input/ch03/time_series_train.csv')
event_history = pd.read_csv('../input/ch03/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------
print(train_x.head())
# event_history 為過去舉辦過的活動資訊，包含日期、活動欄位的 DataFrame

# occurrences 為含有日期、是否舉辦特價的欄位的 DataFrame
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# 透過計算累積和來表現每個日期的累積出現次數
# occurrences 為含有日期、拍賣的累積出現次數的 DataFrame
occurrences['sale'] = occurrences['sale'].cumsum()
print(occurrences.head())
# 以日期為 key 來結合訓練資料
train_x = train_x.merge(occurrences, on='date', how='left')
print(train_x.head())