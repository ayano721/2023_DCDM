import numpy as np
from tensorflow import keras
import os
import tensorflow as tf
import struct
import scipy.sparse as sparse

def AQ(A,mult_precond):
    M = np.zeros(A.shape)
    for i in range(A.shape[0]):
        M[:,i] = mult_precond(A[i])
    return M

def off_diagonal_sum(A):
    return sum(sum(A)) - sum(A.diagonal())

def get_vec(file_rhs,normalize = False, d_type='double'):
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r1 = np.delete(r0, [0])
        if normalize:
            return r1/np.linalg.norm(r1)
        else: 
            return r1            

def get_frame_from_source(n,data_folder_name,normalize = True, d_type='double'):
    file_rhs = data_folder_name + "div_v_star_"+str(n)+".bin"
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        #print(r0[0])
        r0 = np.delete(r0, [0])
        if normalize:
            return r0/np.linalg.norm(r0)
        return r0            
    else:
        print("No file for n = "+str(n)+" in data folder "+data_folder_name)

def readA(dim,filenameA):
    dim2 = dim*dim;
    mat_A = np.zeros((dim2,dim2));
    with open(filenameA, 'rb') as f:
        length = 8;
        b = f.read(length);
        val = struct.unpack('N', b);
        for j in range(val[0]):
            lenght = 8;
            bj = f.read(lenght);
            ele = struct.unpack('N',bj);
            for k in range(ele[0]):
                len_double = 8;
                bk = f.read(len_double);
                elejk = struct.unpack('d',bk);
                mat_A[j][k] = elejk[0];
    return mat_A

def readA_sparse(dim, filenameA, dtype = 'd'):                                                                                                                                                              
    dim2 = dim**3
    cols = []
    outerIdxPtr = []
    rows = []
    if dtype == 'd':
        len_data = 8        
    elif dtype == 'f':
        len_data = 4     
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        #print("nnz = ",nnz)
        #print("num_rows = ", num_rows)
        #print("num_cols = ",num_cols)
        #print("outS = ", outS)
        #print("innS = ", innS)
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            #length = 8
            b = f.read(len_data)
            data[i] = struct.unpack(dtype, b)[0]
            
        for i in range(outS):
            length = 4
            b = f.read(length)
            outerIdxPtr[i] = struct.unpack('i', b)[0]
            
        for i in range(nnz):
            length = 4
            b = f.read(length)
            cols[i] = struct.unpack('i', b)[0]
     
    #print(num_rows, num_cols, nnz, outS, innS) 
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        #print(ii," ; ",outerIdxPtr[ii])
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
     
     
    #print(nnz,len(rows),len(cols),len(data))
    #print(outerIdxPtr[num_rows-1])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])

def readA_sparse_from_bin(dim, filenameA, dtype = 'd'):
    dim2 = dim**3
    cols = []
    outerIdxPtr = []
    rows = []
    if dtype == 'd':
        len_data = 8
    elif dtype == 'f':
        len_data = 4    
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        #print("nnz = ",nnz)
        #print("num_rows = ", num_rows)
        #print("num_cols = ",num_cols)
        #print("outS = ", outS)
        #print("innS = ", innS)
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            #length = 8
            b = f.read(len_data)
            data[i] = struct.unpack(dtype, b)[0]
            
        for i in range(outS):
            length = 4
            b = f.read(length)
            outerIdxPtr[i] = struct.unpack('i', b)[0]
            
        for i in range(nnz):
            length = 4
            b = f.read(length)
            cols[i] = struct.unpack('i', b)[0]
     
    print(num_rows, dim2, nnz, outS) 
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        #print(ii," ; ",outerIdxPtr[ii])
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
     
     
    #print(nnz,len(rows),len(cols),len(data))
    #print(outerIdxPtr[num_rows-1])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])
     
def ReadEigenSpMat(dim, filenameA):
    cols = []
    outerIdxPtr = []
    data = []
    rows = []
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        print("nnz = ",nnz)
        print("num_rows = ", num_rows)
        print("num_cols = ",num_cols)
        print("outS = ", outS)
        print("innS = ", innS)
        
        for i in range(nnz):
            length = 8
            b = f.read(length)
            data = data + [struct.unpack('d', b)[0]]
            
        for i in range(outS):
            length = 4
            b = f.read(length)
            outerIdxPtr = outerIdxPtr + [struct.unpack('i', b)[0]]
            
        for i in range(nnz):
            length = 4
            b = f.read(length)
            cols = cols + [struct.unpack('i', b)[0]]
            
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows = rows + [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])

def ReadEigenSpMat(dim, filenameA):
    cols = []
    outerIdxPtr = []
    data = []
    rows = []
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        print("nnz = ",nnz)
        print("num_rows = ", num_rows)
        print("num_cols = ",num_cols)
        print("outS = ", outS)
        print("innS = ", innS)
        for i in range(nnz):
            length = 8 
            b = f.read(length)
            data = data + [struct.unpack('d', b)[0]]
                
        for i in range(outS):
            length = 4 
            b = f.read(length)
            outerIdxPtr = outerIdxPtr + [struct.unpack('i', b)[0]]
                
        for i in range(nnz):
            length = 4 
            b = f.read(length)
            cols = cols + [struct.unpack('i', b)[0]]
                
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows = rows + [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    return sparse.csr_matrix((data, (rows, cols)),[dim,dim])

def load_model_from_source(model_file_source):
    json_file = open(model_file_source + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ml_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    ml_model.load_weights(model_file_source + "model.h5")
    print("Loaded model from disk")
    return ml_model

    
    
    
    
    
    



