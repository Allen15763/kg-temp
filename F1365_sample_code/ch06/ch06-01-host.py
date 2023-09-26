import numpy as np

# 參數 1 和參數 2 的候選
param1_list = [3, 5, 7, 9]
param2_list = [1, 2, 3, 4, 5]

# 使用 Grid search 搜索的參數組合
grid_search_params = []
for p1 in param1_list:
    for p2 in param2_list:
        grid_search_params.append((p1, p2))
        
print('grid_search_params:', grid_search_params)

"""優化
法1:
grid_search_params = [(p1, p2) for p1 in param1_list for p2 in param2_list]
法2:
import itertools
import collections

# 参数 1 和参数 2 的候选
param1_list = [3, 5, 7, 9]
param2_list = [1, 2, 3, 4, 5]

# 使用 itertools.product 的類生成所有参数组合(tuple)
params_product = itertools.product(param1_list, param2_list)


# 使用 collections.namedtuple 命名元组
SearchParams = collections.namedtuple('SearchParams', ['param1', 'param2'])

# 将每个参数组合转换为命名元组，并存储在列表中
grid_search_params = [SearchParams(*params) for params in params_product]

# 調用
grid_search_params[2].param1, grid_search_params[2].param2
"""

# 使用 Random search 搜索的參數組合。可能有重複組合
random_search_params = []
trials = 15
for i in range(trials):
    p1 = np.random.choice(param1_list)
    p2 = np.random.choice(param2_list)
    random_search_params.append((p1, p2))

print('random_search_params:', random_search_params)

"""
from collections import Counter
counter = Counter(random_search_params)

duplicates = [item for item, count in counter.items() if count > 1]
print(f'duplicated items: {duplicates}')
"""