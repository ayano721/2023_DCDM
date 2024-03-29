# This is an example test code for the paper 
# The test code solves the linear system Ax = b, where A is the system matrix, and b is the velocity divergence
# Both A and b is creted from a simulation. We provide various simulations, such as smoke plume, smoke passing bunny 

#%% Load the required libraries
import sys
import os    
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np 
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import argparse

import conjugate_gradient as cg
import helper_functions as hf


#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128, 256],
                    help="N or resolution of test", default = 64 )
parser.add_argument("-k", "--trained_model_type", type=int, choices=[64, 128],
                    help="which model to test", default=64)
parser.add_argument("-f", "--float_type", type=int, choices=[16, 32],
                    help="model parameters' float type", default=32)
parser.add_argument("-ex", "--example_type", type=str, choices=["smoke_plume", "smoke_passing_bunny"],
                    help="example type", default="smoke_passing_bunny")
parser.add_argument("-fn", "--frame_number", type=int,
                    help="which frame in sim to test", default=10)
parser.add_argument("--max_cg_iter", type=int,
                    help="maximum cg iteration", default=1000)
parser.add_argument("-tol","--tolerance", type=float,
                    help="tolerance for both DCDM and CG algorithm", default=1.0e-4)
parser.add_argument("--verbose_dcdm", type=bool,
                    help="prints residuals of DCDM algorithm for each iteration", default=False)
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="path to the dataset")
parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str,
                    help="Determines if DCDM uses GPU. Default DCDM only uses CPU.", default="")
parser.add_argument("--num_vectors_deflated_pcg", type=int,
                    help="number of vectors used in DeflatedPCG", default=16)
parser.add_argument('--skip_dcdm', action="store_true", 
                    help='skips dcdm tests')
parser.add_argument('--skip_icpcg', action="store_true", 
                    help='skips icpcg test')
parser.add_argument('--skip_deflated_pcg', action="store_true", 
                    help='skips deflated pcg test')
parser.add_argument('--skip_cg', action="store_true", 
                    help='skips cg test')

args = parser.parse_args()


#%%
N = args.resolution

k = args.trained_model_type

if N == 64:
    print("For N=64 resolution there is only 64-trained model.")
    k=64    

float_type = args.float_type
if float_type == 16:
    dtype_ = tf.float16
if float_type == 32:
    dtype_ = tf.float32

example_name = args.example_type

frame_number = args.frame_number

max_cg_iter = args.max_cg_iter

tol = args.tolerance 

verbose_dcdm = args.verbose_dcdm

verbose_icpcg = verbose_dcdm

dataset_path = args.dataset_dir

# This determines if DCDM uses only CPU or uses GPU. By default it only uses CPU.
os.environ["CUDA_VISIBLE_DEVICES"]= args.CUDA_VISIBLE_DEVICES

#%% 
if example_name in ["smoke_plume", "smoke_passing_bunny"]:
    matrix_frame_number = 1
else:
    matrix_frame_number = frame_number
    
trained_model_name = dataset_path + "/trained_models/model_N"+str(N)+"_from"+str(k)+"_F"+str(float_type)+"/"

#%% Getting RHS for the Testing
d_type='double'
def get_vector_from_source(file_rhs,d_type='double'):
    if(os.path.exists(file_rhs)):
        return_vector = np.fromfile(file_rhs, dtype=d_type)
        return_vector = np.delete(return_vector, [0])
        return return_vector
    else:
        print("RHS does not exist at "+ file_rhs)
        
print("Matrix A and rhs b is loading...")
initial_normalization = False 
b_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name+"/div_v_star"+str(matrix_frame_number)+".bin" 
A_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name+"/matrixA_"+str(matrix_frame_number)+".bin" 
A = hf.readA_sparse(N, A_file_name,'f')
b = get_vector_from_source(b_file_name)
CG = cg.ConjugateGradientSparse(A)
normalize_ = False 

#%% Testing
if not args.skip_dcdm:
    print("Loading the model...")
    trained_model_name = dataset_path + "/trained_models/model_N"+str(N)+"_from"+str(k)+"_F"+str(float_type)+"/"
    model = hf.load_model_from_source(trained_model_name)
    model.summary()
    print("Model loaded. Number of parameters in the model is ",model.count_params())
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=dtype_),training=False).numpy()[0,:,:].reshape([N**3]) 
    # Dummy Calling
    model_predict(b)
    
    print("DCDM is running...")
    t0=time.time()                                  
    max_dcdm_iter = 100                                                                                                                                      
    x_sol, res_arr= CG.dcdm(b, np.zeros(b.shape), model_predict, max_dcdm_iter, tol, False ,verbose_dcdm)
    time_cg_ml = time.time() - t0
    print("DCDM took ", time_cg_ml," secs.")

if not args.skip_cg:
    print("CG is running...")
    t0=time.time()
    x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,max_cg_iter,tol,True)
    time_cg = time.time() - t0
    print("CG took ",time_cg, " secs")

if not args.skip_deflated_pcg:
    print("DeflatedPCG is running")
    t0=time.time()
    x_sol_cg, res_arr_cg = CG.deflated_pcg(b, max_cg_iter, tol, args.num_vectors_deflated_pcg, True)
    time_cg = time.time() - t0
    print("Deflated PCG took ",time_cg, " secs")

if not args.skip_icpcg:
    b2 = b[A.getnnz(1)>0]
    A2 = A[A.getnnz(1)>0]
    A2 = A2[:,A.getnnz(0)>0]
    CG2 = cg.ConjugateGradientSparse(A2)

    icpcg_test_folder = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+ example_name +"/L"+str(matrix_frame_number)+".npz"
    L = sparse.load_npz(icpcg_test_folder)
    
    def ic_precond(x):    
        y_inter = sparse.linalg.spsolve_triangular(L,x, lower=True) #Forward sub                                            
        return sparse.linalg.spsolve_triangular(L.transpose(),y_inter, lower=False) #backward sub
    
    print("IncompleteCholeskyPCG is running...")
    t0 = time.time()
    tol_icpcg = np.linalg.norm(b2)*tol
    x_sol, res_arr_icpcg = CG2.pcg_normal(np.zeros(b2.shape),b2,ic_precond,max_cg_iter,tol_icpcg,True)
    time_icpcg = time.time() - t0
    print("IncompleteCholeskyPCG took ", time_icpcg, " secs")

