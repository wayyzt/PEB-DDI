# PEB-DDI: A Task-Specific Dual-View Substructural Learning Framework for Drug--Drug Interaction Prediction

## Introduction

Adverse drug-drug interactions (DDIs) pose potential risks in polypharmacy due to unknown physicochemical incompatibilities between co-administered drugs. Recent studies have utilized multi-layer graph neural network architectures to model hierarchical molecular substructures of drugs, achieving excellent DDI prediction performance. While extant substructural frameworks effectively encode interactions from atom-level features, they overlook valuable chemical bond representations within molecular graphs. More critically, given the multifaceted nature of DDI prediction tasks involving both known and novel drug combinations, previous methods lack tailored strategies to address these distinct scenarios. The resulting lack of adaptability impedes further improvements to model performance. 
To tackle these challenges, we propose PEB-DDI, a DDI prediction learning framework with enhanced substructure extraction. First, the information of chemical bonds is integrated and synchronously updated with the atomic nodes. Then, different dual-view strategies are selected based on whether novel drugs are present in the prediction task. Rigorous evaluations on benchmark datasets underscore PEB-DDI's superior performance. Notably, on DrugBank, it achieves an outstanding accuracy rate of 98.18\% when predicting previously unknown interactions among approved drugs. Even when faced with novel drugs, PEB-DDI demonstrates a notable improvement in accuracy, with relative enhancements of 2.23% and 1.09%.

![PEB-DDI](D:\github\local\image\PEB-DDI.jpg)

# Installation

You can create a virtual environment using conda

```
conda create -n ddi python=3.7
source activate ddi
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-geometric==2.0.3
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c rdkit rdkit
```

## Usage

Change the 'dataset_name' in args.json to evaluate different datasets in transductive (t) and inductive (i) tasks for fold 0/1/2.

```
"dataset_name": "datasetName_t/i_0/1/2",
```

Train the model for transductive task

```
python transductive_train.py
```

Evaluate the model for transductive task with existing pkl files.

```
python transductive_test.py
```

Train the model for inductive task

```
python inductive_s12.py
```

Evaluate the model for inductive task with existing pkl files.

```
python inductive_test.py
```



