# IIITH final project (2023)


## Statement

The problem of question answering involves developing natural language processing models that can accurately locate the correct answer from a given context and provide a natural language response to a given question. The challenge of traditional question answering models is that they struggle to answer questions where the answer is not explicitly stated in the context. To address this issue, there is a need for models that can effectively leverage contextual information and provide accurate answers to both traditional answerable contextual questions and difficult no-answer questions. Additionally, existing models may have limitations in capturing long-term dependencies and achieving parallelization, which can impact their speed and accuracy. The problem statement is to develop effective question answering models that can overcome these challenges and improve reading comprehension.

## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code

4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You can directly run script shell/train_bidaf.sh or shell/train_qanet.sh to train the model and save checkpoints

6. For test the following checkpoint is available https://drive.google.com/file/d/17pAJfAQrEIiDfaXhFjYTEF-3Jyr1sXiV/view?usp=share_link place it under ../save/train/model-01 or modify code as per your specification and run shell/test_ensemble.sh

## References 

1. Kim, Yoon (2014) Convolutional Neural Networks for Sentence Classification
(https://arxiv.org/pdf/1408.5882v2.pdf)
2. Seo, Minjoon & Kembhavi, Aniruddha & Farhadi, Ali & Hajishirzi, Hananneh (2016) Bidirectional Attention Flow for Machine Comprehension
(https://arxiv.org/pdf/1611.01603.pdf)
3. https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py 
4. https://github.com/BangLiu/QANet-PyTorch 
5. https://github.com/allenai/bi-att-flow
