# DCDM


This repository is based on the paper [under review].
We accelerate fluid simulations by embedding a neural network in an existing solver for pressure, replacing an expensive pressure projection linked to a Poisson equation on a computational fluid grid that is usually solved with iterative methods (CG or Jacobi methods). 
We implemented our code with TensorFlow (keras) for training parts.

## Requirements
* Python 3.8
* keras 2.8.0
* tensorflow 2.3.0
* CUDA 11

We are using virtual environments using conda.

## Setting Up Environment for Running DCDM

1. Create and activate a conda environement:
```
conda create --name venvname[name of vertual env] python=3.8
conda activate venvname
```

2. Install tensorflow. Conda should install keras, numpy, and scipy automatically. If not, install them using conda.
```
conda install tensorflow
```

3. Download the data file **[here](https://www.dropbox.com/s/dlhvuyub87i9cyl/icml2023data.tar.gz?dl=0)** to the ```project source directory``` and extract all data files.
```
tar -zxvf icml2023data.tar.gz
cd icml2023data
tar -zxvf original_matA.tar.gz
tar -zxvf test_matrices_and_vectors.tar.gz
tar -zxvf trained_models.tar.gz
```
Once all the data are extracted, the files structures should look like the following:
```
.
└── (Project Source Directory)
    ├── src
    └── icml2023data
        ├── test_matrices_and_vectors  
        └── trained_models
        └── original_matA
```

* `test_matrices_and_vectors` contains the RHS vectors and system matrices for different examples with different grid resolutions (N = 64, 128, 256).
  * RHS (b) of the linear system (Ax = b): `div_v_star_[frame].bin`
  * System matrix data: `matrixA_[frame].bin` (stored in compressed sparse row format).
* `trained_models` includes pre-trained models. Each model is trained using a particular resolution `K`, and is designed for a particular simulation resolution `N`. Models are correspondingly named model_N[N]_from[K]_F32.

## Running Tests

To compare the performance of different methods (DCDM, CG, DeflatedPCG, ICPCG),
```
cd src/
python3 test_all_cg.py --dataset_dir <dataset_path>
```
Note that dataset are located at `../icml2023data` if users follow the steps in the previous section. To view all the command line options, users can find them using the following command:
```
python3 test_all_cg.py --help
```


## Training
### Dataset
We placed dataset here [place of the folder]. Here, we generated this dataset by calculating Ritz vectors, so we also show our code here. [GeneratingData.py]


### Training model
We placed pre-trained models here: ```/trained_model```. We also placed the training code here.

```
cd ThisDir/src
python train_N64.py [epoch number] [epoch number for saved model] [batch size] [loading data size for once] [gpu usage memory 1024*int] [GPU id]
```