import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import gc
import scipy.sparse as sparse
import time

sys.path.insert(1, '../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

import argparse


#%% Get Arguments from parser
parser = argparse.ArgumentParser()

parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of test", default = 64 )

parser.add_argument("--total_number_of_epochs", type=int,
                    help="Total number of epochs for training", default=1000)

parser.add_argument("--epoch_each_number", type=int,
                    help="epoch number of", default=1)

parser.add_argument("--batch_size", type=int,
                    help="--batch_size.", default=10)

parser.add_argument("--loading_number", type=int,
                    help="loading number of each iteration", default=100)

parser.add_argument("--gpu_usage", type=int,
                    help="gpu usage, in terms of GB.", default=3)

parser.add_argument("--gpu_idx", type=str,
                    help="which gpu to use.", default='1')

parser.add_argument("--data_dir", type=str,
                    help="path to the folder containing dataset vectors", default='../icml2023data/')



args = parser.parse_args()

N = args.resolution
epoch_num = args.total_number_of_epochs
epoch_each_iter = args.epoch_each_number
b_size = args.batch_size
loading_number = args.loading_number
gpu_usage = args.gpu_usage
which_gpu = args.gpu_idx


project_name = "3D_N"+str(N)
project_folder_subname = os.path.basename(os.getcwd())
print("project_folder_subname = ", project_folder_subname)

project_folder_general = "../training/3D_N"+str(N)+"/"


dim2 = N**3
lr = 1.0e-4

# you can modify gpu memory usage editing here
os.environ["CUDA_VISIBLE_DEVICES"]=which_gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=gpu_usage)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


if N == 64:
    A_file_name = args.data_dir + "/original_matA/A_origN"+str(N)+".bin"  
elif N == 128:
    A_file_name = args.data_dir + "/original_matA/A_oriN"+str(N)+".bin"  

A_sparse_scipy = hf.readA_sparse(N, A_file_name,'f')


CG = cg.ConjugateGradientSparse(A_sparse_scipy)

coo = A_sparse_scipy.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
A_sparse = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)

def custom_loss_function_cnn_1d_fast(y_true,y_pred):
    b_size_ = len(y_true)
    err = 0
    for i in range(b_size):
        A_tilde_inv = 1/tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.sparse.sparse_dense_matmul(A_sparse, tf.reshape(y_pred[i],[dim2,1])),axes=1)
        qTb = tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.reshape(y_true[i],[dim2,1]), axes=1)
        x_initial_guesses = tf.reshape(y_pred[i],[dim2,1]) * qTb * A_tilde_inv
        err = err + tf.reduce_sum(tf.math.square(tf.reshape(y_true[i],[dim2,1]) - tf.sparse.sparse_dense_matmul(A_sparse, x_initial_guesses)))
    return err/b_size_

#%% Training model 
dim = N
fil_num=16
input_rhs = keras.Input(shape=(dim, dim, dim, 1))
first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

apa = layers.AveragePooling3D((2, 2,2), padding='same')(lb) 
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

upa = layers.UpSampling3D((2, 2,2))(apa) + lb
upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

last_layer = layers.Dense(1, activation='linear')(upa)

model = keras.Model(input_rhs, last_layer)
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast) 
model.optimizer.lr = lr;
model.summary()


training_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_training_loss.npy"
validation_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_validation_loss.npy"
training_loss = []
validation_loss = []


# if you want to use your own dataset, you can change here.
if N == 64:
    foldername = "../datasets/N64/b_rhs_20000_10000_ritz_vectors_newA_90_10_random_N64/"
elif N == 128:
    foldername = "../datasets/N128/b_rhs_20000_10000_ritz_vectors_newA_90_10_random_N128/"

total_data_points = 20000
for_loading_number = round(total_data_points/loading_number)
b_rhs = np.zeros([loading_number,dim2])

for i in range(1,epoch_num):    
    print("Training at i = " + str(i))
    
    training_loss_inner = []
    validation_loss_inner = []
    t0=time.time()    
    perm = np.random.permutation(total_data_points)
    for ii in range(for_loading_number):
        print("Sub_training at ",ii,"/",for_loading_number," at training ",i)

        # Loasing the data
        for j in range(loading_number):
            with open(foldername+str(perm[loading_number*ii+j])+'.npy', 'rb') as f:  
                b_rhs[j] = np.load(f)
        
        sub_train_size = round(0.9*loading_number)
        sub_test_size = loading_number - sub_train_size
        iiln = ii*loading_number
        x_train = tf.convert_to_tensor(b_rhs[0:loading_number].reshape([loading_number,dim,dim,dim,1]),dtype=tf.float32) 
        x_test = tf.convert_to_tensor(b_rhs[sub_train_size:loading_number].reshape([sub_test_size,dim,dim,dim,1]),dtype=tf.float32)         
         
        hist = model.fit(x_train,x_train,
                        epochs=epoch_each_iter,
                        batch_size=b_size,
                        shuffle=True,
                        validation_data=(x_test,x_test))
        
        training_loss_inner = training_loss_inner + hist.history['loss']
        validation_loss_inner = validation_loss_inner + hist.history['val_loss']  
    
    time_cg_ml = (time.time() - t0)
    print("Training loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Validation loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Time for epoch = ",i," is ", time_cg_ml)
    training_loss = training_loss + [sum(validation_loss_inner)/for_loading_number]
    validation_loss = validation_loss + [sum(validation_loss_inner)/for_loading_number]
    
    os.makedirs(name = project_folder_general+project_folder_subname+ "/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i))
    os.system("touch "+project_folder_general+project_folder_subname+ "/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/model.json")
    model_json = model.to_json()
    model_name_json = project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/"
    with open(model_name_json+ "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name_json + "model.h5")
    
    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))
    print(training_loss)
    print(validation_loss)
