# DCDM


This repository is based on the paper [under review].
We accelerate fluid simulations by embedding a neural network in an existing solver for pressure, replacing an expensive pressure projection linked to a Poisson equation on a computational fluid grid that is usually solved with iterative methods (CG or Jacobi methods). 
We implemented our code with TensorFlow (keras) for training parts.

## Requirements
* Python 3.8
* keras 2.8.0
* tensorflow 2.3.0

We are using virtual environments using conda.

## Training
### Dataset
We placed dataset here [place of the folder]. Here, we generated this dataset by calculating Ritz vectors, so we also show our code here. [GeneratingData.py]


### Training model
We placed pre-trained models here: ```/trainedmodel```. We also placed the training code here.

```
cd ThisDir/src
python train_N64.py [epoch number] [epoch number for saved model] [batch size] [loading data size for once] [gpu usage memory 1024*int] [GPU id]
```



## Running test
The test data is under ```dataset_mlpcg/test_matrices_and_vectors ``` folder.The place of the folder is shown as below.
* div_v_star_"frame".bin --> RHS of Ax = b 
* matrixA_"frame".bin --> system matrix data, they are stored in compressed sparse row (CSR) format.



## All datasets
From here, you can donwload the ```trained model```, ```testing matrix``` and ```A^(0,0) matrix data for generating datasets for training model```.

https://www.dropbox.com/s/1t989hxfobg4i89/DCDM2023.zip?dl=0



```
.
└── HomeDir(cloned Dir)
    ├── src
    └── dataset_mlpcg (please donwload from above link, you can place anywhere and just set the pass)
        ├── test_matrices_and_vectors  
        └── trained_models
        └── original_matA
            
```

* test_matrices_and_vectors : they includes the rhs and system matrix A for examples for each dimension.(N64, N128, N256)
* trained_models : they includes trained model for each resolution. For each resolution, we have 2 type of trained over resolution models at ```64``` and ```128```.
At ```dim``` dimension and trained over ```K``` resolution model are named 'model_N[dim]_from[K]_FN32.

## Installation

1. Extract  this folder, download the required dataset from above link and place it.

2. Create and activate a conda environement
```
conda create --name venvname[name of vertual env] python=3.8
conda activate venvname
```

3. Install keras, tensorflow with cuda_11.7 and the rest of necessary packages from the requirements file.
To install the rest of the packages running these command.
```
pip install -r requirements.txt
```


