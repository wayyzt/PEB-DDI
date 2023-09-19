import pandas as pd
import os
import sys
os.chdir(sys.path[0])

# 读取三个CSV文件
train = pd.read_csv('./DeepDDI_train_id.csv')
val = pd.read_csv('./DeepDDI_valid_id.csv')
test = pd.read_csv('./DeepDDI_test_id.csv')

# 筛选出标签为1的数据
# filtered_train = train[train['label'] == 1]
# filtered_val = val[val['label'] == 1]
# filtered_test = test[test['label'] == 1]

# 将筛选出的数据合并
# ddis = pd.concat([filtered_train, filtered_val, filtered_test])
ddis = pd.concat([train, val, test])


# 保存合并的数据到新的CSV文件
# filtered_train.to_csv('train_zhang.csv', index=False)
# filtered_val.to_csv('val_zhang.csv', index=False)
# filtered_test.to_csv('test_zhang.csv', index=False)
ddis.to_csv('ddis_deep.csv', index=False)