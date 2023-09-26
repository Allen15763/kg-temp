import numpy as np
import pandas as pd

# -----------------------------------
# 結合兩份表格資料
# -----------------------------------
# 讀取資料
train = pd.read_csv('../input/ch03/multi_table_train.csv')
product_master = pd.read_csv('../input/ch03/multi_table_product.csv')
user_log = pd.read_csv('../input/ch03/multi_table_log.csv')

# -----------------------------------
# 假設一個如上圖所示的資料框架
# train         : 訓練資料（含使用者 ID, 商品 ID, 變數等欄位）
# product_master: 商品清單（含商品 ID 和商品資訊等欄位）
# user_log      : 使用者活動的記錄檔資料（含使用者 ID 和各種活動資訊等欄位）

# 合併商品清單和訓練資料
train = train.merge(product_master, on='product_id', how='left')

# 統計每個使用者活動的記錄檔行數，並和訓練資料結合
user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'})
train = train.merge(user_log_agg, on='user_id', how='left')
