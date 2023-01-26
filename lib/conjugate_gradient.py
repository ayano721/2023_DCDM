import numpy as np
import tensorflow as tf
import scipy
import time
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc
import numpy.linalg as LA
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import inv
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse import csr_matrix

class ConjugateGradientSparse:
    def __init__(self, A_sparse):
        if A_sparse.shape[0] != A_sparse.shape[1]:
            print("A is not a square matrix!")
        self.n = A_sparse.shape[0]
        self.machine_tol = 1.0e-17;
        A_sp = A_sparse.copy()
        self.A_sparse = A_sp.astype(np.float32)
    
    # here x is np.array with dimension n
    def multiply_A(self,x):
        #return np.matmul(self.A,x)
        return self.A_sparse.dot(x)

    # here x is np.array with dimension n
    def multiply_A_sparse(self,x):
        #return np.matmul(self.A,x)
        return self.A_sparse.dot(x)
    
    def norm(self,x):
        return np.linalg.norm(x)

    def dot(self,x,y):
        return np.dot(x,y)
    
    
    def multi_norm(self,xs):
        norms = np.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            norms[i] = self.norm(xs[i])
        return norms
    

    
    def multi_dot(self,xs,ys):
        dots = np.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            dots[i] = self.dot(xs[i],ys[i])
        return dots
    
    def multi_dot_V2(self,xs,ys):
        return np.einsum('ij,ij->i',xs,ys)

    def multi_dot_and_scale(self,xs,ys,scales):
        dots = np.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            dots[i] = self.dot(xs[i],ys[i])/scales[i]
        return dots
    
    def multi_normalize(self, rs):
        rs_normalized = rs.copy()
        r_norms = self.multi_norm(rs)
        for i in range(rs.shape[0]):
            rs_normalized[i] = rs_normalized[i]/r_norms[i]
        return rs_normalized
    
    def get_zero_rows(self):
        return np.where(self.A_sparse.getnnz(1)==0)[0]
    
    def get_nonzero_rows(self):
        zero_rows = self.get_zero_rows()
        nonzero_rows = list(range(self.n))
        for i in range(len(zero_rows)):
            nonzero_rows.remove(zero_rows[i])
        return np.array(nonzero_rows)
 
    
    #create_approximate_eigenmodes (old name)
    def create_ritz_vectors(self, b, num_vectors, sorting=True):
        W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_vectors, 1.0e-12)
        #W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_vectors, 1.0e-12)
        if(num_vectors != len(diagonal)):
            print("Careful. Lanczos Iteration converged too early, num_vectors = "+str(num_vectors)+" > "+str(len(diagonal)))
            num_vectors = len(diagonal)
        tri_diag = np.zeros([num_vectors,num_vectors])
        for i in range(1,num_vectors-1):
            tri_diag[i,i] = diagonal[i]
            tri_diag[i,i+1] = sub_diagonal[i]
            tri_diag[i,i-1]= sub_diagonal[i-1]
        tri_diag[0,0]=diagonal[0]
        tri_diag[0,1]=sub_diagonal[0]
        tri_diag[num_vectors-1,num_vectors-1]=diagonal[num_vectors-1]
        tri_diag[num_vectors-1,num_vectors-2]=sub_diagonal[num_vectors-2]
        eigvals,Q0 = np.linalg.eigh(tri_diag)
        eigvals = np.real(eigvals)
        Q0 = np.real(Q0)
        Q1 = np.matmul(W.transpose(),Q0).transpose()
        if sorting:
            Q = np.zeros([num_vectors,self.n])
            sorted_eig_vals = sorted(range(num_vectors), key=lambda k: -eigvals[k])
            for i in range(num_vectors):
                Q[i]=Q1[sorted_eig_vals[i]].copy()
            return Q
        else:
            return Q1
          
    #precond is num_vectors x self.n np array
    #Old name: create_lambda_vals

    def create_ritz_values(self,Q,relative=True):
        lambda_ = np.zeros(Q.shape[0])
        for i in range(Q.shape[0]):
            dotx = np.dot(Q[i],Q[i])
            if dotx < self.machine_tol:
                print("Error! Zero vector in matrix Q.")
                return
            lambda_[i] = np.dot(Q[i],self.multiply_A_sparse(Q[i]))
            if relative:
                lambda_[i] = lambda_[i]/dotx
        return lambda_
        


    def mult_diag_precond(self,x):
        y = np.zeros(x.shape)
        for i in range(self.n):
            if self.A_sparse[i,i]>self.machine_tol:
                y[i] = x[i]/self.A_sparse[i,i]
        return y

    def mult_precond_method1(self,x,Q,lambda_):
        y = np.copy(x)
        for i in range(Q.shape[0]):
            qTx = np.dot(Q[i],x)
            y = y + qTx*(1/lambda_[i] - 1.0)*Q[i]
        return y

    def Q_Ac_Qt_y(self,x,Q,lambda_):
        y = np.copy(x)
        Q_s = Q.astype(np.float32)
        Qb = (Q_s@x).astype(np.float32)
        #calc QAinv
        diag_lam = np.diag(lambda_)
        QtAinv = Q_s.transpose()@LA.inv(diag_lam)
        y = QtAinv@Qb
        return y
    
    
    
    #update this one
    def pcg_normal(self, x_init, b, mult_precond, max_it=100, tol=1.0e-12,verbose=True):
        #b is rhs
        #x_init is initial prediction
        #mult_precond is a function for multiplying preconditioner
        res_arr = []
        x = x_init.copy()
        ax = self.multiply_A_sparse(x_init)
        res = self.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]

        r = b.copy()
        r = r - ax
        z = mult_precond(r)        
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz
        for it in range(max_it):
            ax = self.multiply_A_sparse(p)
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            z = mult_precond(r)        
            rz = np.dot(r,z)
            res = self.norm(self.multiply_A_sparse(x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            if it != max_it - 1: 
                beta = rz/rz_k
                pk_1 = p.copy()
                p = z.copy()
                p = p + pk_1*beta
                rz_k = rz
        
        if verbose:
            print("PCG converged in "+str(max_it)+ " iterations to the final residual = "+str(res))
        return [x, res_arr]
     

        
    def pcg_normal_old(self, x_init, b, mult_precond, max_it=100, tol=1.0e-12,verbose=True):
        #b is rhs
        #x_init is initial prediction
        #mult_precond is a function for multiplying preconditioner
        res_arr = []
        x = x_init.copy()
        ax = np.matmul(self.A, x_init)
        res = np.linalg.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        r = b.copy()
        r = r - ax
        #lambda_ = self.create_lambda_vals(Q)
        #z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_)
        z = mult_precond(r)        
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz
        for it in range(max_it):
            ax = np.matmul(self.A, p)
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            #z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_)
            z = mult_precond(r)        
            rz = np.dot(r,z)
            res = np.linalg.norm(np.matmul(self.A, x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            beta = rz/rz_k
            pk_1 = p.copy()
            p = z.copy()
            p = p + pk_1*beta
            rz_k = rz
        
        if verbose:
            print("PCG residual = "+str(res))
            print("PCG converged in "+str(max_it)+ " iterations.")
            
        return [x, res_arr]
    

    
    def cg_normal(self,x,b,max_it=100,tol=1.0e-12,verbose=False):
        #res = np.linalg.norm(self.multiply_A(x)-b)
        #res = self.norm
        ax = self.multiply_A_sparse(x)
        r = b.copy()
        r = r - ax
        res = self.norm(r)
        res_arr = [res]
        if verbose:
            print("first cg residual is "+str(res))
        if res < tol:
            if verbose:
                print("CG converged in 0 iterations")
            return [x, res_arr]
        p = r.copy()
        rr_k = np.dot(r,r)
        for it in range(max_it):
            ax = self.multiply_A_sparse(p)
            alpha = rr_k/np.dot(p,ax)
            x = x+alpha*p
            res = self.norm(self.multiply_A_sparse(x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("CG converged in "+str(it)+" iteration to the residual "+str(res))
                return [x, res_arr]
            #r = r - ax*alpha
            r = b - self.multiply_A(x)
            rr_k1 = np.dot(r,r)
            beta = rr_k1/rr_k
            q = p.copy()
            p = r.copy()
            p = p+ q*beta
            rr_k = rr_k1
        
        if verbose:
            print("CG used max = "+str(max_it)+" iteration to the residual "+str(res))  
        
        return [x, res_arr]
    
    def cg_normal_test2(self,x,b,max_it=100,tol=1.0e-12,verbose=False):
        #res = np.linalg.norm(self.multiply_A(x)-b)
        #res = self.norm
        ax = self.multiply_A_sparse(x)
        r = b.copy()
        r = r - ax
        res = self.norm(r)
        res_arr = [res]
        if verbose:
            print("first cg residual is "+str(res))
        if res < tol:
            if verbose:
                print("CG converged in 0 iterations")
            return [x, res_arr]
        p = r.copy()
        rr_k = np.dot(r,r)
        zero_idxs = self.get_zero_rows()
        zero_eigvec = np.ones([self.n])
        zero_eigvec[zero_idxs] = 0.0
        zero_eigvec = zero_eigvec/self.norm(zero_eigvec)
        for it in range(max_it):
            ax = self.multiply_A_sparse(p)
            alpha = rr_k/np.dot(p,ax)
            x = x+alpha*p
            res = self.norm(self.multiply_A_sparse(x)-b)
            print(it,res)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("CG converged in "+str(it)+" iteration to the residual "+str(res))
                return [x, res_arr]
            #r = r - ax*alpha
            r = b - self.multiply_A(x)
            r[zero_idxs] = 0.0
            r = r - zero_eigvec*self.dot(r, zero_eigvec)
            rr_k1 = np.dot(r,r)
            beta = rr_k1/rr_k
            q = p.copy()
            p = r.copy()
            p = p+ q*beta
            rr_k = rr_k1
        
        if verbose:
            print("CG used max = "+str(max_it)+" iteration to the residual "+str(res))  
        
        return [x, res_arr]
    
    def cg_normal_test(self,x,b,max_it=100,tol=1.0e-12,verbose=False):
        #res = np.linalg.norm(self.multiply_A(x)-b)
        #res = self.norm
        ax = self.multiply_A_sparse(x)
        r = b.copy()
        r = r - ax
        res = self.norm(r)
        res_arr = [res]
        if verbose:
            print("first cg residual is "+str(res))
        if res < tol:
            if verbose:
                print("CG converged in 0 iterations")
            return [x, res_arr]
        p = r.copy()
        rr_k = np.dot(r,r)
        for it in range(max_it):
            ax = self.multiply_A_sparse(p)
            alpha = rr_k/np.dot(p,ax)
            x = x+alpha*p
            res = self.norm(self.multiply_A_sparse(x)-b)
            print(it,res)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("CG converged in "+str(it)+" iteration to the residual "+str(res))
                    
                return [x, res_arr]
            r = r - ax*alpha
            #r = b - self.multiply_A(x)
            rr_k1 = np.dot(r,r)
            beta = rr_k1/rr_k
            q = p.copy()
            p = r.copy()
            p = p+ q*beta
            rr_k = rr_k1
        
        if verbose:
            print("CG used max = "+str(max_it)+" iteration to the residual "+str(res))  
        return [x, res_arr]
    
    def lanczos_pcg(self, x_init, b, mult_precond, max_it=100, tol=1.0e-12,verbose=True):
        x_sol = x_init.copy()
        r = b.copy() - self.multiply_A(x_init)
        
        q_bar0 = np.zeros(b.shape) #q_-1
        q_bar1 = mult_precond(r)   #q_0
        t1 = np.sqrt(np.dot(r,q_bar1))
        q_bar1 = q_bar1/t1
        
        Aq_bar1 = self.multiply_A(q_bar1)
        beta1 = 0 #beta0 = 0
        alpha1 = np.dot(q_bar1,Aq_bar1) 
        d1 = alpha1 #d0 = alpha0
        p_bar1 = q_bar1/d1
        x_sol = x_sol + t1*p_bar1
        
        for i in range(1, max_it):
            q_bar2 = mult_precond(Aq_bar1) # q_bar2 = MAq_bar1
            beta2 = np.sqrt(np.dot(q_bar2,Aq_bar1) - alpha1**2-beta1**2) #beta1 
            q_bar2 = (q_bar2 - alpha1*q_bar1 - beta1*q_bar0)/beta2
            Aq_bar2 = self.multiply_A(q_bar2)
            alpha2 = np.dot(Aq_bar2, q_bar2) #alpha1
            mu = beta2/d1
            d2 = alpha2 - d1*mu**2
            
            
            alpha1 = alpha2
            beta1 = beta2
            d1 = d2
            q_bar0 = q_bar1.copy()
            q_bar1 = q_bar2.copy()
            Aq_bar1 = Aq_bar2.copy()
        
        return x_sol        
    
    def lanczos_iteration(self, b, max_it=10, tol=1.0e-10):
        Q = np.zeros([max_it, len(b)])
        if max_it==1:
            Q[0]=b.copy()
            Q[0]=Q[0]/self.norm(Q[0])
            return Q, [np.dot(Q[0],self.multiply_A_sparse(Q[0]))], []
        if max_it<=0:
            print("CG.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ",self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        #norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy()/norm_b
        #Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        #sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            Q[it+1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])
            Q[it+1] = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            sub_diagonal[it] = self.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))
        
        return Q, diagonal, sub_diagonal
    
    def lanczos_iteration_with_normalization_correction(self, b, max_it=10, tol=1.0e-10):
        Q = np.zeros([max_it, len(b)])
        if max_it==1:
            Q[0]=b.copy()
            Q[0]=Q[0]/self.norm(Q[0])
            return Q, [np.dot(Q[0],self.multiply_A_sparse(Q[0]))], []
        if max_it<=0:
            print("CG.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ",self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        #norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy()/norm_b
        #Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        #sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            Q[it+1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])            
            v = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            for j in range(it-1):
                v = v - Q[j]*self.dot(v, Q[j])
            Q[it+1] = v.copy()
            sub_diagonal[it] = self.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))
        
        return Q, diagonal, sub_diagonal
    
    def lanczos_iteration_with_normalization_correctionV2(self, b, normalization_number, max_it=10, tol=1.0e-10):
        Q = np.zeros([max_it, len(b)])
        if max_it==1:
            Q[0]=b.copy()
            Q[0]=Q[0]/self.norm(Q[0])
            return Q, [np.dot(Q[0],self.multiply_A_sparse(Q[0]))], []
        if max_it<=0:
            print("CG.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ",self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        #norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy()/norm_b
        #Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        #sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            if it%50==0:
                print("Lanczoz it = ",it)
            #Q[it+1] = np.matmul(self.A, Q[it])
            Q[it+1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])            
            v = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            for j in range(min(normalization_number,it-1)):
                v = v - Q[it-2-j]*self.dot(v, Q[it-2-j])
            Q[it+1] = v.copy()
            sub_diagonal[it] = self.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))
        
        return Q, diagonal, sub_diagonal
    
#from numba import njit, prange
#    @njit(parallel=True)
    def lanczos_iteration_with_normalization_correction_parallel(self, b, max_it=10, tol=1.0e-10):
        Q = np.zeros([max_it, len(b)])
        if max_it==1:
            Q[0]=b.copy()
            Q[0]=Q[0]/self.norm(Q[0])
            return Q, [np.dot(Q[0],self.multiply_A_sparse(Q[0]))], []
        if max_it<=0:
            print("CG.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ",self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        #norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy()/norm_b
        #Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        #sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            if it%100==0:
                print("Lanczoz iteration at = ",it)
            #Q[it+1] = np.matmul(self.A, Q[it])
            Q[it+1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])            
            v = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            for j in range(it-1):
                vQj = self.dot(v, Q[j])
                for jj in range(v.shape[1]):
                    v[jj] = v[jj] - Q[j,jj]*vQj
            Q[it+1] = v.copy()
            sub_diagonal[it] = self.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))
        
        return Q, diagonal, sub_diagonal


    def deflated_pcg_n(self, b,max_it = 100,tol = 1.0e-15,num_vectors = 16, verbose = False):
        res_arr = [] 
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        Q = self.lanczos_iteration(b_iter,num_vectors)
        Q = self.create_ritz_vectors(b_iter,num_vectors)
        lambda_ = (self.create_ritz_values(Q))
        diag_lam = diags(lambda_)
        A_c_inv = ( inv(diag_lam)).tocsr()
        Q_sp = csr_matrix(Q,dtype=np.float32)
        gc.collect()
        zAcinvz = csr_matrix(Q_sp.transpose()*(A_c_inv)*(Q_sp))#.tocsr() 
        x = x_init.copy()
        ax = self.multiply_A_sparse(x_init)
        res = self.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        mult_precond = lambda x_in_val: self.mult_diag_precond(x_in_val)
        
        r = b_iter #- ax(0)
        x = zAcinvz.dot(r)
        ax =  self.multiply_A_sparse(x)
        r = b_iter -  ax
        z = mult_precond(r)  #precond z0
        z0 = z.copy()
        az = self.multiply_A_sparse(z0)
        #new p0
        p = z0 -  zAcinvz.dot(az)

        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ap = self.multiply_A_sparse(p)
            alpha = rz_k/np.dot(p,ap)
            x = x + p*alpha
            r = r - ap*alpha
            #after updated x and r
            #UPDATING r
            z = mult_precond(r)             
            rz = np.dot(r,z)
            res = self.norm(self.multiply_A_sparse(x)-b)
            res_arr = res_arr + [res]
            print("PCG residual = "+str(res)+ ", Ite" + str(it))
            if res < tol: 
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            if it != max_it - 1: 
                az = self.multiply_A_sparse(z)
                if abs(rz_k)>0:
                  beta = rz/rz_k
                else: 
                  beta = 0
                print(beta,":beta")
                pk_1 = p.copy()
                #p = z.copy()
                p = z + pk_1*beta - zAcinvz.dot(az)
                rz_k = rz 
         
        if verbose:
            print("PCG converged in "+str(max_it)+ " iterations to the final residual = "+str(res))
        return [x, res_arr]


    def deflated_pcg(self, b,max_it = 100,tol = 1.0e-15,num_vectors = 16, verbose = False):
        res_arr = [] 
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        Q = self.create_ritz_vectors(b_iter,num_vectors)
        lambda_ = (self.create_ritz_values(Q))
        #x0 r0
        x = x_init.copy()
        ax = self.multiply_A_sparse(x_init)
        res = self.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        mult_precond = lambda x_in_val: self.mult_diag_precond(x_in_val)
        
        r = b_iter #- ax(0)

        x = self.Q_Ac_Qt_y(r,Q,lambda_)  #precond z0
        ax =  self.multiply_A_sparse(x)
        
        r = b_iter -  ax
        z = mult_precond(r)  #precond z0
        z0 = z.copy()
        az = self.multiply_A_sparse(z0)#.astype(np.float32)
        #new p0
        tempv = self.Q_Ac_Qt_y(az,Q,lambda_) 
        p = z0 -  tempv
        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ap = self.multiply_A_sparse(p)
            alpha = rz_k/np.dot(p,ap)
            x = x + p*alpha
            r = r - ap*alpha
            #after updated x and r
            #UPDATING r
            z = mult_precond(r)             
            rz = np.dot(r,z)
            res = self.norm(self.multiply_A_sparse(x)-b)
            res_arr = res_arr + [res]
            if res < tol: 
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            if it != max_it - 1: 
                az = self.multiply_A_sparse(z)#.astype(np.float32)
                if abs(rz_k)>0:
                  beta = rz/rz_k
                else: 
                  beta = 0
                pk_1 = p.copy()
                tempv = self.Q_Ac_Qt_y(az,Q,lambda_) 
                p = z + pk_1*beta - tempv#zAcinvz.dot(az)
                rz_k = rz 
         
        if verbose:
            print("PCG converged in "+str(max_it)+ " iterations to the final residual = "+str(res))
        return [x, res_arr]

    def restarted_pcg_automatic(self, b, max_outer_it = 100, pcg_inner_it = 1, tol = 1.0e-15, method = "approximate_eigenmodes", num_vectors = 16, verbose = False):
        res_arr = []    
        x_sol = np.zeros(b.shape)
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        for i in range(max_outer_it):
            if method == "approximate_eigenmodes":
                time_sta = time.time()
                Q = self.create_ritz_vectors(b_iter,num_vectors)
                lambda_ = self.create_ritz_values(Q)
                time_end = time.time()
                tim = time_end - time_sta
                print("time  ; ",tim)
                mult_precond = lambda x: self.mult_precond_method1(x,Q,lambda_)
            else:           
                print("Method is not recognized!")
                return  
            x_sol1, res_arr1 = self.pcg_normal(x_init, b_iter, mult_precond, pcg_inner_it, tol, False)
            x_sol = x_sol + x_sol1
            b_iter = b - self.multiply_A_sparse(x_sol)

            b_norm = np.linalg.norm(b_iter)
            res_arr = res_arr + res_arr1[0:pcg_inner_it]                
            print("restarting at i = "+ str(i)+ " , residual = "+ str(b_norm))                 
            if b_norm < tol:
                print("RestartedPCG converged in "+str(i)+" iterations.")                                                                                                                            
                break
        return x_sol, res_arr


    def restarted_pcg_manual(self, b, mult_precond_method, max_outer_it = 100, pcg_inner_it = 1, tol = 1.0e-15, verbose = False):
        #mult_precond_method(CG,x, b)
        res_arr = []
        x_sol = np.zeros(b.shape)
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        for i in range(max_outer_it):
            mult_precond = lambda x: mult_precond_method(self, x, b_iter)
            x_sol1, res_arr1 = self.pcg_normal(x_init, b_iter, mult_precond, pcg_inner_it, tol, False)
            x_sol = x_sol + x_sol1
            b_iter = b - self.multiply_A_sparse(x_sol)
            b_norm = np.linalg.norm(b_iter)
            res_arr = res_arr + res_arr1[0:pcg_inner_it]
            if verbose:
                print("restarting at i = "+ str(i)+ " , residual = "+ str(b_norm))            
            if b_norm < tol:
                print("RestartedPCG converged in "+str(i)+" iterations to the residual "+str(res_arr[-1]))
                break
        return x_sol, res_arr
    
    
    #LDLT
    def forward_subs(self,L, b):
        #here L is sparse matrix        
        #y=[]
        #y = np.zeros(b.shape)
        y = b.copy()
        for i in range(len(b)):
            if(abs(L[i,i])>1.0e-5):   
                #y.append(b[i])
                rr , cc = L.getrow(i).nonzero()
                for j in cc:
                    if j < i:
                        y[i]=y[i]-(L[i,j]*y[j])
                    
                y[i]=y[i]/L[i,i]
        return y

    def back_subs(self,U,y):
        x=np.zeros_like(y)
        U2 = sparse.triu(U,k=1,format="csr")
        
        for i in range(len(x),0,-1):
            if(abs(U[i-1,i-1])>1.0e-5):
                #uu = np.dot(U[i-1,i:],x[i:])
                #ui  = U.getrow(i).dot(x)[0]
                #uu = U2.dot(x)[0]
                x[i-1]=(y[i-1]-U.getrow(i-1).dot(x)[0])/U[i-1,i-1]
        return x


    
    def ldlt(self):
        #L = self.A.copy()
        L = sparse.tril(self.A_sparse,k=0,format="csr")
        #rows_L,  cols_L = L.nonzero()
        #L = np.matrix(np.zeros((n,n)))
        #D = np.matrix(np.zeros((n,n)))    
        D = sparse.diags(self.A_sparse.diagonal(),format="csr")                              
        L[0,0] = 1.0;                  
        #A1=A@A            
        for i in range(1,self.n):
            #if (abs(L[i,i])> 1.0e-5):
            #print("i = ",i)
            rr , cc = L.getrow(i).nonzero()
            #print("cc = ",cc)
            for j in cc:  #for j in range(i)        0,1,...,i-1        
                if j<i:
                    lld = self.A_sparse[i,j]
                    for k in cc: #careful
                        if k<j:
                            lld = lld - L[i,k]*L[j,k]*D[k,k]
                    if abs(D[j,j])>1.0e-5:
                        L[i,j] = lld/D[j,j]
            
            ld = self.A_sparse[i,i]              
            for k in cc:
                if k < i:
                    ld = ld - L[i,k]*L[i,k]*D[k,k]       
            D[i,i] = ld
            L[i,i] = 1.0
        return L, D
 
    
    def ldlt_pcg(self,L, D, b, max_it = 100, tol = 1.0e-15,verbose = False):
        res_arr = []                   
        x_sol = np.zeros(b.shape)
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        #print("L and D are being computed...")
        #L,D = self.ldlt()
        #print("L and D are computed.")
        U = D@L.T
        def mult_precond_ldlt(x):                                                       
            y_inter=self.forward_subs(L,x)
            y=self.back_subs(U,y_inter) 
            return y                 
        mult_precond = lambda x: mult_precond_ldlt(x)
        x_sol, res_arr = self.pcg_normal(x_init,b_iter,mult_precond,max_it,tol,verbose)
        return x_sol, res_arr
    
    def gauss_seidel(self, b, x, max_iterations, tolerance, verbose):
        #x is the initial condition
        iter1 = 0
        res_arr = []
        #Iterate
        for k in range(max_iterations):
            iter1 = iter1 + 1
            #print ("The solution vector in iteration", iter1, "is:", x) 
            res = np.linalg.norm(b-np.matmul(self.A,x))
            res_arr = res_arr+[res]
            if verbose:
                print(k, res)
            x_old  = x.copy()
            
            #Loop over rows
            for i in range(self.A.shape[0]):
                if self.A[i,i] > self.machine_tol:    
                    x[i] = (b[i] - np.dot(self.A[i,:i], x[:i]) - np.dot(self.A[i,(i+1):], x_old[(i+1):])) / self.A[i ,i]
        return x, res_arr


    def create_lower_and_upper_matrices(self):
        zero_rows = self.get_zero_rows()
        #for i in zero_rows:
        #    self.A_sparse[i,i]=1.0
        #self.L = sparse.tril(self.A_sparse, k=0, format=None)
        #self.U = sparse.triu(self.A_sparse, k=1, format=None)    
        self.L = sparse.csr_matrix(sparse.tril(self.A_sparse, k=0, format=None))
        self.U = sparse.csr_matrix(sparse.triu(self.A_sparse, k=1, format=None))
        for i in zero_rows:
            self.L[i,i]=1.0

    def gauss_seidel_sparse(self, b, x_init, max_iterations=100, tol=1.0e-4, verbose=False):
        #x is the initial condition
        #self.create_lower_and_upper_matrices()        
        res_arr = []
        x = x_init.copy()
        zero_rows = self.get_zero_rows()
        nonzero_rows = self.get_nonzero_rows()
        for i in zero_rows:
            if abs(b[i]) >1.0e-12:
                print("Gauss-Seidel Error: b is not in the column space.")
                return x, res_arr
        #Iterate
        for k in range(max_iterations):
            r = b - self.U.dot(x)
            x = sparse.linalg.spsolve_triangular(self.L, r)
            norm_r = np.linalg.norm(b-self.multiply_A(x))
            res_arr = res_arr+[norm_r]

            if norm_r<tol:
                print("Gauss-Seidel method converged to residual ",norm_r, " in ",k, " iterations.")
                return x, res_arr
            if verbose:
                print(k, norm_r)
                
        print("Gauss-Seidel method converged to residual ",norm_r, " in ",max_iterations, " iterations.")
        return x, res_arr
    
    def cg_on_ML_generated_subspaceFN(self, b, x_init, model_predict, max_it=100,tol=1e-10,fluid = False,verbose=True):
        dim2 =len(b)
        x_sol = np.zeros(b.shape)
        res_arr = []
        p0 = np.zeros(dim2)
        p1 = np.zeros(dim2)
        Ap0 = np.zeros(dim2)
        Ap1 = np.zeros(dim2)
        alpha0 = 1.0
        alpha1 = 1.0
        r = b - self.multiply_A(x_init)
        norm_r = self.norm(r)
        res_arr = [norm_r]
        tol = norm_r*tol
        if verbose:
            print("Initial residual =",norm_r)
        if norm_r<tol:
            print("cg_on_ML_generated_subspace converged in 0 iterations to residual ",norm_r)
            return x_init, res_arr
        
        x_sol = x_init.copy()
        for i in range(max_it):
            r_normalized = r/norm_r
            t0 = time.time()     
            if fluid == False:
              q = model_predict(r_normalized)
            else:
              q = model_predict(r)
            t1 = time.time()
            print('Cal time:{}'.format(t1-t0))
            q = q - p1*self.dot(q, Ap1)/alpha1 - p0*self.dot(q, Ap0)/alpha0
            Ap0 = Ap1.copy()
            Ap1 = self.multiply_A(q) 
            p0 = p1.copy()
            p1 = q.copy()
            alpha0 = alpha1
            alpha1 = self.dot(p1, Ap1)
            beta = self.dot(p1,r)
            x_sol = x_sol + p1*beta/alpha1
            r = r - Ap1*beta/alpha1
            norm_r = self.norm(r)           
            r = b - self.multiply_A(x_sol) 
            norm_r = self.norm(r)           
            res_arr = res_arr + [norm_r]
            if verbose:
                print(i+1,norm_r)
            if norm_r < tol:
                print("cg_on_ML_generated_subspace converged in ", i+1, " iterations to residual ",norm_r)
                print("Actual norm = ",self.norm(b-self.multiply_A(x_sol)))
                return x_sol,res_arr
            
        print("cg_on_ML_generated_subspace converged in ", max_it, " iterations to residual ",norm_r)
        print("Real norm = ",self.norm(b-self.multiply_A(x_sol)))
        return x_sol,res_arr    


    def dcdm(self, b, x_init, model_predict, max_it=100,tol=1e-10,fluid = False,verbose=True):
        dim2 =len(b)
        x_sol = np.zeros(b.shape)
        res_arr = []
        p0 = np.zeros(dim2)
        p1 = np.zeros(dim2)
        Ap0 = np.zeros(dim2)
        Ap1 = np.zeros(dim2)
        alpha0 = 1.0
        alpha1 = 1.0
        r = b - self.multiply_A(x_init)
        norm_r = self.norm(r)
        res_arr = [norm_r]
        tol = norm_r*tol
        if verbose:
            print("Initial residual =",norm_r)
        if norm_r<tol:
            print("DCDM converged in 0 iterations to residual ",norm_r)
            return x_init, res_arr
        
        x_sol = x_init.copy()
        for i in range(max_it):
            r_normalized = r/norm_r
            if fluid == False:
              q = model_predict(r_normalized)
            else:
              q = model_predict(r)
            q = q - p1*self.dot(q, Ap1)/alpha1 - p0*self.dot(q, Ap0)/alpha0
            Ap0 = Ap1.copy()
            Ap1 = self.multiply_A(q) #!!!!
            p0 = p1.copy()
            p1 = q.copy()
            alpha0 = alpha1
            alpha1 = self.dot(p1, Ap1)
            beta = self.dot(p1,r)
            x_sol = x_sol + p1*beta/alpha1
            r = r - Ap1*beta/alpha1
            norm_r = self.norm(r)           
            r = b - self.multiply_A(x_sol) 
            norm_r = self.norm(r)           
            res_arr = res_arr + [norm_r]
            if verbose:
                print(i+1,norm_r)
            if norm_r < tol:
                print("DCDM converged in ", i+1, " iterations to residual ",norm_r)
                return x_sol, res_arr
            
        print("DCDM converged in ", max_it, "(maximum iteration) iterations to residual ",norm_r)
        return x_sol, res_arr    



    def cg_on_ML_generated_subspace_withoutAouth(self, b, x_init, model_predict, max_it=100,tol=1e-10,fluid = False,verbose=True):
        dim2 =len(b)
        x_sol = np.zeros(b.shape)
        res_arr = []
        p0 = np.zeros(dim2)
        p1 = np.zeros(dim2)
        Ap0 = np.zeros(dim2)
        Ap1 = np.zeros(dim2)
        alpha0 = 1.0
        alpha1 = 1.0
        r = b - self.multiply_A(x_init)
        norm_r = self.norm(r)
        res_arr = [norm_r]
        tol = norm_r*tol
        if verbose:
            print("Initial residual =",norm_r)
        if norm_r<tol:
            print("cg_on_ML_generated_subspace converged in 0 iterations to residual ",norm_r)
            return x_init, res_arr
        
        x_sol = x_init.copy()
        for i in range(max_it):
            r_normalized = r/norm_r
            q = model_predict(r_normalized)
            Ap0 = Ap1.copy()
            Ap1 = self.multiply_A(q) #!!!!
            p0 = p1.copy()
            p1 = q.copy()
            alpha0 = alpha1
            alpha1 = self.dot(p1, Ap1)
            beta = self.dot(p1,r)
            x_sol = x_sol + p1*beta/alpha1
            r = r - Ap1*beta/alpha1
            norm_r = self.norm(r)           
            r = b - self.multiply_A(x_sol) 
            norm_r = self.norm(r)           
            res_arr = res_arr + [norm_r]
            if verbose:
                print(i+1,norm_r)
            if norm_r < tol:
                print("cg_on_ML_generated_subspace converged in ", i+1, " iterations to residual ",norm_r)
                print("Actual norm = ",self.norm(b-self.multiply_A(x_sol)))
                return x_sol,res_arr
            
        print("cg_on_ML_generated_subspace converged in ", max_it, " iterations to residual ",norm_r)
        print("Real norm = ",self.norm(b-self.multiply_A(x_sol)))
    
    def cg_on_ML_generated_subspace_A_normal_general(self, b, x_init, model_predict, orthonormalization_num=2, max_it=100,tol=1e-10,verbose=True, true_norm_calculation = False, output_search_directions=False):
        dim2 =len(b)
        k = orthonormalization_num
        x_sol = np.zeros(b.shape)
        res_arr = []
        P_temp = np.zeros([max(k,1), dim2])
        AP_temp = np.zeros([max(k,1), dim2])
        if output_search_directions:
            P_memory = np.zeros([max_it, dim2])
        alphas = np.ones([max(k,1)])                            

        r = b - self.multiply_A(x_init)
        norm_r = self.norm(r)
        res_arr = [norm_r]
        tol = norm_r*tol
        if verbose:
            print("Initial residual =",norm_r)
        if norm_r<tol:
            print("cg_on_ML_generated_subspace converged in 0 iterations to residual ",norm_r)
            return x_init, res_arr
        
        x_sol = x_init.copy()
        for i in range(max_it):
            q = model_predict(r/norm_r)
            for j in range(k):
                q = q - P_temp[j]*self.dot(q, AP_temp[j])/alphas[j]
            if output_search_directions:
                P_memory[i] = q.copy()
            
            if k!=0:
                ii = i%k
            else:
                ii=0
            P_temp[ii]=q.copy()
            AP_temp[ii] = self.multiply_A(q)
            alphas[ii] = self.dot(q, AP_temp[ii])
            beta = self.dot(q,r)
            x_sol = x_sol + q*beta/alphas[ii]
            if true_norm_calculation:
                r = b - self.multiply_A(x_sol)
            else:
                r = r - AP_temp[ii]*beta/alphas[ii]
            norm_r = self.norm(r)           
            res_arr = res_arr + [norm_r]
            if verbose:
                print(i+1,norm_r)
            if norm_r < tol:
                print("cg_on_ML_generated_subspace converged in ", i+1, " iterations to residual ",norm_r)
                print("Actual norm = ",self.norm(b-self.multiply_A(x_sol)))
                if output_search_directions:
                    return x_sol,res_arr, P_memory
                else:
                    return x_sol,res_arr
            
        print("cg_on_ML_generated_subspace converged in ", max_it, " iterations to residual ",norm_r)
        print("Real norm = ",self.norm(b-self.multiply_A(x_sol)))
        if output_search_directions:
            return x_sol,res_arr, P_memory
        else:
            return x_sol,res_arr

         
  
  
   
   


   
           
   
