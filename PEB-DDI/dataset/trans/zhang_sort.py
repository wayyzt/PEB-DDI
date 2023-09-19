import pandas as pd
import os
import sys
os.chdir(sys.path[0])
# 加载数据集
df = pd.read_csv('zhang/ZhangDDI_test.csv')

# 根据'label'列排序
df_sorted = df.sort_values(by='label', ascending=False)

# 保存排序后的数据集
df_sorted.to_csv('zhang/ZhangDDI_test_sorted.csv', index=False)

