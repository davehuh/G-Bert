# G-Bert
Pre-training of Graph Augmented Transformers for Medication Recommendation

## Intro
G-Bert combined the power of **G**raph Neural Networks and **BERT** (Bidirectional Encoder Representations from Transformers) for medical code representation and medication recommendation. We use the graph neural networks (GNNs) to represent the structure information of medical codes from a medical ontology. Then we integrate the GNN representation into a transformer-based visit encoder and pre-train it on single-visit EHR data. The pre-trained visit encoder and representation can be fine-tuned for downstream medical prediction tasks. Our model is the first to bring the language model pre-training schema into the healthcare domain and it achieved state-of-the-art performance on the medication recommendation task.

## Environment Setup to Reproduce the Experiment
- Clone the GAMENet repository to retrieve required pkl and mapping artifacts: https://github.com/davehuh/GAMENet
- Clone the repository
- Move data from MIMIC-III and GAMENet. See Data Requirement section.
- Create a Python Virtual Environment. I'm using Python 3.9. Refer to Virtual Environment Setup Section below.
- Install Visual Studio 2019 Community and C++ Redistribution v14.0. https://visualstudio.microsoft.com/vs/older-downloads/
- Install Cuda 11.3 https://developer.nvidia.com/cuda-11.3.0-download-archive
- Pip install all the requirements. See the Pip section.
- Edit torch_geometric/utils/scatter.py. See scatter instruction below.
- run run_alternative.sh in /code/
- Takes around ~6 hours to fully cycle 15 iterations on Nvidia GTX 1080 Ti GPU

## Data Requirements
Move the following artifacts to /data/
- ehr_adj.pkl
- RENAME data_final.pkl to data_gamenet.pkl
- drug-atc.csv
- ndc2atc_level4.csv
- ndc2rxnorm_mapping.txt
- data-multi-side.pkl
- drug-DDI.csv
- PRESCRIPTIONS.csv
- DIAGNOSES_ICD.csv

## Cuda Install
After installing the Cuda 11.3 toolkit, make sure you see it installed in shell.
```shell
nvcc --version
```

## Python Virtual Environment Setup (use git bash if on Windows)
```shell
which python
python3 -m venv PATH/TO/venv_gbert
```
cd to venv_gbert
```shell
cd Scripts
source activate
```

## PIP Install
While the virtual environment is active in your shell environment, execute the following pip commands.
```shell
python -m pip install --upgrade pip
pip install wheel
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install dill
pip install jupyter
pip install pandas
pip install tqdm
pip install tensorboardX
pip install sklearn
pip install torch_geometric==1.0.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## Source code modification: torch_geometric
cd to venv_gbert
```shell
cd /venv_gbert/Lib/site-packages/torch_geometric/utils
```
edit scatter.py
```
add from torch_scatter import scatter
replace: out = op(src, index, 0, None, dim_size, fill_value)
to:  out = scatter(src, index, 0, None, dim_size)
save and exit
```

## Requirements
- python 3.9 >=
- Cuda enabled GPU
- pytorch>=0.4
- python>=3.5
- torch_geometric==1.0.3

## Guide
We list the structure of this repo as follows:
```latex
.
├── [4.0K]  code/
│   ├── [ 13K]  bert_models.py % transformer models
│   ├── [5.9K]  build_tree.py % build ontology
│   ├── [4.3K]  config.py % hyperparameters for G-Bert
│   ├── [ 11K]  graph_models.py % GAT models
│   ├── [   0]  __init__.py
│   ├── [9.8K]  predictive_models.py % G-Bert models
│   ├── [ 721]  run_alternative.sh % script to train G-Bert
│   ├── [ 19K]  run_gbert.py % fine tune G-Bert
│   ├── [ 19K]  run_gbert_side.py
│   ├── [ 18K]  run_pretraining.py % pre-train G-Bert
│   ├── [4.4K]  run_tsne.py # output % save embedding for tsne visualization
│   └── [4.7K]  utils.py
├── [4.0K]  data/
│   ├── [4.9M]  data-multi-side.pkl 
│   ├── [3.6M]  data-multi-visit.pkl % patients data with multi-visit
│   ├── [4.3M]  data-single-visit.pkl % patients data with singe-visit
│   ├── [ 11K]  dx-vocab-multi.txt % diagnosis codes vocabulary in multi-visit data
│   ├── [ 11K]  dx-vocab.txt % diagnosis codes vocabulary in all data
│   ├── [ 29K]  EDA.ipynb % jupyter version to preprocess data
│   ├── [ 18K]  EDA.py % python version to preprocess data
│   ├── [6.2K]  eval-id.txt % validation data ids
│   ├── [6.9K]  px-vocab-multi.txt % procedure codes vocabulary in multi-visit data
│   ├── [ 725]  rx-vocab-multi.txt % medication codes vocabulary in multi-visit data
│   ├── [2.6K]  rx-vocab.txt % medication codes vocabulary in all data
│   ├── [6.2K]  test-id.txt % test data ids
│   └── [ 23K]  train-id.txt % train data ids
└── [4.0K]  saved/
    └── [4.0K]  GBert-predict/ % model files to reproduce our result
        ├── [ 371]  bert_config.json 
        └── [ 12M]  pytorch_model.bin
```
## Cite 

Please cite our paper if you find this code helpful:

```
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
```

## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.
- [GAMENet](https://github.com/sjy1203/GAMENet)
- [Bert_HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT)
- [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)


