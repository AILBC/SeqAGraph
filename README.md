# Enhanced Template-Free Reaction Prediction with Molecular Graphs and Sequence-based Data Augmentation

Here is the code for *"Enhanced Template-Free Reaction Prediction with Molecular Graphs and Sequence-based Data Augmentation"*

***The manuscript is under submission now, please cite this page if you find our work is helpful***

## 1. Overview
The overview of the directory is shown as:
```
├─index_elemtwise---------------#the python package for custom kernel
├─index_elemtwise_cuda----------#the cpp and cuda source code for custom index_select + elemtwise operate
│  ├─csrc
│  │  └─cuda
│  └─index_elemtwise
├─model-------------------------#the py source for our model
│  ├─ckpt
│  │  └─Model0
│  │      ├─50k
│  │      ├─50k_class
│  │      ├─full
│  │      └─mit
│  ├─inference------------------#the py source for beam search (some of them are from huggingface)
│  └─preprocess-----------------#the data preprocess source for our model
│      └─data
│          ├─uspto_50k
│          │  ├─process
│          │  └─raw
│          ├─uspto_full
│          │  ├─process
│          │  └─raw
│          └─uspto_MIT
│              ├─process
│              └─raw
└─scripts-----------------------#the .sh file for quick start
```

## 2. Environment setup
Code was run and evaluated for:

    - python 3.10.9
    - pytorch 2.0.0 (for SDPA kernel)
    - torch-scatter 2.1.1+pt20cu117
    - rdkit 2022.03.2

Models were trained on RTX A5000 with 24GB memory for larger batch size(e.g. 64\*2), which also available for less GPU memory with an appropriate batch size setting and larger gradient accumulation steps(e.g. 32\*2 and accumulate 4 steps for 6GB).

Note that an different version of rdkit may result in different SMILES canonicalization results.

## 3. Custom indexing kernel
The custom CUDA kernel for the operation `src1.index_select(0, idx1) ~ src2.index_select(0, idx2)` is now available, which is one of the operation in our padding-free global attention for molecular graphs, it is faster than naive pytorch operation and also support AMP in pytorch. It should works well in most of the situation and will be further optimized later. You can install it by running :
```
python index_elemtwise_cuda/setup.py install
```

## 4. Data preprocessing
We mainly use **USPTO-50k**, **USPTO-full** and **USPTO-MIT** datasets for training and evaluation, you can download them manually at the following address:
```
uspto_50k  https://www.dropbox.com/scl/fo/4b4vlp3muns4hsp0sqp5q/h?dl=0&rlkey=272fqtrlk57jkmom6i5h0xomu
uspto_MIT  https://www.dropbox.com/scl/fo/kkny008b93tgi7to2030s/h?dl=0&rlkey=fo5fykax0rc6d9oi9czg9bqe9
uspto_full https://www.dropbox.com/scl/fo/tljb4n130cj2e3hq9wjqz/h?dl=0&rlkey=vwtmxeum4989017rn86eqiq07
```
notice that **USPTO-50k** are already available in the source file.

After downloading them, please put the `data.zip` into `model/preprocess/data/${dataset_name}/raw`, and then run the following scripts for preprocessing:
```
scripts/preprocess.sh -> uspto_50k
scripts/preprocess_MIT.sh -> uspto_MIT
scripts/preprocess_full.sh -> uspto_full
```

## 5. Training
You can train the model by:
```
scripts/train.sh -> uspto_50k
scripts/train_class.sh -> uspto_50k with reaction class
scripts/train_MIT.sh -> uspto_MIT
scripts/train_full.sh -> uspto_full
```

The log file, checkpoint queue list, tensorboard file, and checkpoints are available at `model/ckpt/Model0/${training_start_time}`, notice that the model will eval automatically and generate average checkpoint according to the checkpoint queue, like `AVG_0.pt`, `AVG_1.pt` and so on, for the large dataset like uspto_full, it will use a random subset to run the evaluation during training.

## 6. Evaling
The checkpoints are available at the following address, which includes `.pt` file, log file, and evaluation results in the manuscript:
```
uspto_50k       https://www.dropbox.com/scl/fo/erf9bkp8rz0ehq6ru43kz/h?dl=0&rlkey=2t6i17z55i7c4hpke05y8skpi
usoto_50k_class https://www.dropbox.com/scl/fo/96h1mb9vkytbb6rem75sr/h?dl=0&rlkey=n5x5lajanrsb9ypvxd5fyfs6s
uspto_MIT       https://www.dropbox.com/scl/fo/pst2lrfk1jw3h7pv47upr/h?dl=0&rlkey=xg4y26yjvp07gp49ay8zfmvbp
uspto_full      https://www.dropbox.com/scl/fo/z5x86bgxxrp9n2jfr3b2u/h?dl=0&rlkey=dozqz4rlh42243w1excfz2ryt
```

After downloading them, please unzip and put the `AVG_MAIN.pt` into `model/ckpt/Model0/${dataset_name}`, and then run the following scripts for evaling and testing:
```
scripts/eval.sh -> uspto_50k
scripts/eval_class.sh -> uspto_50k with reaction class
scripts/eval_MIT.sh -> uspto_MIT
scripts/eval_full.sh -> uspto_full
```

You can find the result in `model/ckpt/Model0/${dataset_name}`, which includes the top-10 accuracy and top-10 invalid rate, or try to use different searching hyperparameters like temperature(T), top-k sampling, top-p sampling, and group beam search.
