import pandas as pd
import numpy as np
import sys
import os
os.chdir(sys.path[0])

# 加载数据
drugs = pd.read_csv("trans/deep/drug_list_deep.csv")
interactions = pd.read_csv("induc/deep/ddis_deep.csv")

# 从已知药物列表中取得所有药物
all_drugs = drugs["drugbank_id"].unique()

# 随机选择20%的药物作为未知药物
unknown_drugs = np.random.choice(all_drugs, size=int(len(all_drugs) * 0.2), replace=False)

# 划分已知药物和未知药物
known_drugs = np.setdiff1d(all_drugs, unknown_drugs)

# 创建训练集，只包含已知药物的DDI
train = interactions[interactions["drugbank_id_1"].isin(known_drugs) & interactions["drugbank_id_2"].isin(known_drugs)]

# 创建测试集的两种划分，S1和S2
test_S1 = interactions[interactions["drugbank_id_1"].isin(unknown_drugs) & interactions["drugbank_id_2"].isin(unknown_drugs)]
test_S2 = interactions[((interactions["drugbank_id_1"].isin(known_drugs) & interactions["drugbank_id_2"].isin(unknown_drugs)) | 
                       (interactions["drugbank_id_1"].isin(unknown_drugs) & interactions["drugbank_id_2"].isin(known_drugs)))]

# 保存结果
train.to_csv("induc/deep/fold1/train.csv", index=False)
test_S1.to_csv("induc/deep/fold1/test_S1.csv", index=False)
test_S2.to_csv("induc/deep/fold1/test_S2.csv", index=False)

# 将unknown_drugs转换为DataFrame并保存为CSV文件
pd.DataFrame(unknown_drugs, columns=["drugbank_id"]).to_csv("induc/deep/fold1/unknown_drugs.csv", index=False)
pd.DataFrame(known_drugs, columns=["drugbank_id"]).to_csv("induc/deep/fold1/known_drugs.csv", index=False)

