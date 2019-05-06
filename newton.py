# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:12:31 2019

@author: Aditya
"""

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.io as sio
import sys

import matplotlib.pyplot as plt

import time
#import pyamg
import os
import multiprocessing as mp
import functools

#from enum import Enum     # for enum34, or the stdlib version

#PMethod= Enum('PMethod', 'given')
#nrowMethod = Enum("nrowMethod", "given")
'''
def permuate(A, rhs, pMethod, pInput = None):
    if(pMethod == 'identity'):
        return(A, rhs, pMethod, pInput)
    elif(pMethod == 'given'):        
        if(len(pInput) != terminal_level - current_level + 1):
            raise ValueError("Needs to be consistent")
        else:
            A = A[:,pInput[0][0]]
            A = A[pInput[0][1], :]
            rhs = rhs[pInput[0][1], :]
            pInput = pInput[1:]
'''    

'''Taken from https://gist.github.com/vtraag/8b82e10e57d93eacc524'''
def permute_sparse_matrix(M, orderRow, orderCol):
    M2 = M.tocsr()
    M2 = M2[:, orderCol]
    M2 = M2[orderRow, :]
    return M2

    

def multigrid(A, rhs, current_level, terminal_level,
              pMethod = 'identity', nMethod = 'given', pInput = None, nInput = None ):

 #   assert(PMethod[pMethod])
 #   assert(nrowMethod[nMethod])

    # TODO: Should check if it's part of the enum     
    if(pMethod == 'given'):
        if(len(pInput) != terminal_level - current_level):
            raise ValueError("Needs to be consistent")
        else:
           # plt.spy(A)
           # plt.show()
            A = permute_sparse_matrix(A, pInput[0][0], pInput[0][1])
            #plt.spy(A)
            #plt.show()
            rhs = rhs[pInput[0][0]]
          #  pInput = pInput[1:]
          
          #TODO Should be a bit more dynamic put will set to identity: 
            pMethod = 'identity'
        
        
    if(nMethod == 'given'):
        if(len(nInput) != terminal_level - current_level + 1):
            raise ValueError("Needs to be consistent")
        else:
            nrows = nInput[0]
            if(terminal_level != current_level):
                nInput = nInput[1:]
    
 #   nrows = get_splits(A)
  #  inds = permuate(A)
#    A = A[:, inds]
 #   A = A[inds, :]
  #  rhs = rhs[inds]
    
    # Split input matrix based on nrows    
    #TODO: Actual algorithm for nrwos.
    A = A.tocsr()
    B0 = A[:nrows, :nrows]
    F0 = A[:nrows, nrows:]
    E0 = A[nrows:, :nrows]
    C0 = A[nrows:, nrows:]
    
    # Same for f0. f0 is the stuff that can be computed independently, 
    # g0 is the stuff that can't be
    f0 = rhs[:nrows]
    g0 = rhs[nrows:]
    
    # On this level, set L0 = I, and U0 = B 
    #TODO: Should this always be the case for non -level 0? 
    
    if(current_level == terminal_level):
        # Direct Solve via ILU
        ILU = sparse.linalg.spilu(B0)
        (L1, U1) = (ILU.L, ILU.U)
        G1 = sparse.linalg.spsolve_triangular(U1.T, (E0.T).todense())
        W1 = sparse.linalg.spsolve_triangular(L1, F0.todense())
        A2 = C0 - G1.T * W1 
        
        # Backsolve 
        f1_prime = sparse.linalg.spsolve_triangular(L1, f0)
        f1_prime = f1_prime.reshape(len(f1_prime), 1)
        inner = G1.T * f1_prime
 #       inner = inner.reshape((len(inner), 1))
        g1_prime = g0 - inner
        
        #More backsolve
        y1 = spsolve(A2, g1_prime)
        y1 = y1.reshape(len(y1), 1)
        u1 = sparse.linalg.spsolve_triangular(U1,(f1_prime.reshape(len(f1_prime), 1) - W1 * y1), False)
        u1 = u1.reshape(len(u1), 1)
        y0 = np.concatenate((u1,y1))
        #Stick them together
        return y0
    else:
        # Descent donwards        
        #Todo: Shouldn't hardcode 9
        L0 = sparse.eye(B0.shape[0])
        U0 = B0
        #Since B0 is diagonal can do this
        inv_U0 = sparse.diags(1/B0.diagonal())
        
        # Use Schur's complement
        inv_L0 = L0
        G0 = E0 * inv_U0;
        W0 = inv_L0 * F0;
        A1 = C0 - G0 * W0;
    
        # Forward/backwards substiution 
        f0_prime = sparse.linalg.spsolve_triangular(L0,f0);
        f0_prime = f0_prime.reshape(len(f0_prime), 1)
        g0_prime = g0 - G0 * f0_prime;

        y0 = multigrid(A1, g0_prime, current_level + 1, terminal_level, 'identity', 'given', pInput, nInput)
        u0 = sparse.linalg.spsolve_triangular(U0,(f0_prime - W0 * y0), False)
        u0 = u0.reshape(len(u0), 1)
        y0 = np.concatenate((u0,y0))
        
        
        if(pInput != None):
            new_col_inds = pInput[0][1]
            y0_reorder = np.zeros_like(y0)
            for i in range(0, len(y0)):
                itemindex = np.where(new_col_inds==i)[0]
                y0_reorder[i] = y0[itemindex]
            y0 = y0_reorder            
        return y0


def run_sample():
    data = sio.loadmat("M_newton")
    A = data["M_newton"]
    data = sio.loadmat("rhs")
    rhs = data["rhs"]

    res = multigrid(A, rhs, 1, 2, 'identity', 'given', None, [32, 8])
    return(res)

def run_sample2():
    model_data = sio.loadmat("5_bus_model")
    y0_data = sio.loadmat("5_bus_model_y0")
    constr_data = sio.loadmat("5_bus_model_const")
    rhs = sio.loadmat('rhs')['rhs']

    M_n = create_M_newton(model_data, y0_data, constr_data)
    (inds, nrows) = find_reordering(model_data['model'])    

    mlSolve =  multigrid(M_n, rhs, 0, 1, 'given','given', [inds], nrows )

    return(mlSolve)


def run_sample3(model_fname, rhs_fname, y0_fname, constr_fname):
    model_data = sio.loadmat(model_fname)
    y0_data = sio.loadmat(y0_fname)
    constr_data = sio.loadmat(constr_fname)
    rhs = sio.loadmat(rhs_fname)['rhs']

    M_n = create_M_newton(model_data, y0_data, constr_data)
    (inds, nrows) = find_reordering(model_data['model'])    
    mlSolve =  multigrid(M_n, rhs, 0, 1, 'given','given', [inds], nrows )
    return(mlSolve)


def get_names(b, n, directory):
    end_name = "%d_nbus%d_ncont" %(b, n)
    model_fname = os.path.join(directory, "model_%s" % end_name)
    rhs_fname = os.path.join(directory, "rhs_%s" % end_name)
    y0_fname = os.path.join(directory, "y0_%s" % end_name)
    constr_fname = os.path.join(directory, "constr_%s" % end_name)
    
    return(model_fname, rhs_fname, y0_fname, constr_fname)


def get_res(A, b, x):
    return(np.linalg.norm((A * x).flatten() - b.flatten()))
    
def run_many():
    
    buses = [5, 189]
    ncont = [1, 2, 3, 4, 5]
    
    timing_dict = {"Multilevel" : {}, "Smooth_AMG" : {}, "direct" : {}}
    residual_dict = {"Multilevel" : {}, "Smooth_AMG" : {}, "direct" : {}}
    for b in buses: 
        for n in ncont:
            end_name = "%d_nbus%d_ncont" %(b, n)
            #Multilevel
            (model_fname, rhs_fname, y0_fname, constr_fname) = get_names(b, n, "Data")
            rhs = sio.loadmat(rhs_fname)['rhs']
#            model_fname, rhs_fname, y0_fname, constr_fname
            
            model_data = sio.loadmat(model_fname)
            y0_data = sio.loadmat(y0_fname)
            constr_data = sio.loadmat(constr_fname)
            #rhs = sio.loadmat(rhs_fname)['rhs']
            M_n = create_M_newton(model_data, y0_data, constr_data)

            start_time = time.time()
            res = run_sample3(model_fname, rhs_fname, y0_fname, constr_fname)
            end_time = time.time() - start_time
            resid = get_res(M_n, rhs, res)
            timing_dict['Multilevel'][end_name] = end_time
            residual_dict['Multilevel'][end_name] = resid
            
            
            #Direct 
            start_time = time.time()
            res = sparse.linalg.lsqr(M_n, rhs)[0]
            end_time = time.time() - start_time
            resid = get_res(M_n, rhs, res)
            
            timing_dict['direct'][end_name] = end_time
            residual_dict['direct'][end_name] = resid


            
            
    return(timing_dict, residual_dict)            

#8 = A
#5= B
#9 = D
#1  = dim


def make_mapping(d):
    key = d.dtype.fields.keys()
    val = np.arange(len(key))
    mapping = {k:v for k, v in zip(key, val)}
    return(mapping)


def get_model_specific(model, cont_num):
    return model[0][0][1][0][cont_num]


    

''' Find the reordeirng - note that this is really just grouping the 
 type of variables together. '''


def get_specific_dim(model, cons_num):
    model_specific = get_model_specific(model, cons_num)
    info_mapping= make_mapping(model_specific)
    dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
    dims = model_specific[0][0][info_mapping['dim']][0][0]
    return((dims, dim_mapping))

def find_reordering(model):
    '''The algorithm here is to permute to get the same type of variables together. 
    Roughtly speaking instead of grouping by contingency directly we want to group by 
    theta variables, lambda variables etc. This somehow gives us a very diagonal 
    B0 matrix which is really easy to work with. 
    '''

    ncont = len(model[0][0][1][0]) - 1
    inds = {'deltaTheta' : [], 'deltaLambda': [], 'deltaNu' : [], 
            'rDual': [], 'rCent' : [], 'rPri' : []}
    
    shift = 0;
    shift2 = 0;
    for cons_num in range(0, ncont + 1):
        model_specific = get_model_specific(model, cons_num)
        info_mapping= make_mapping(model_specific)
        dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
        dims = model_specific[0][0][info_mapping['dim']][0][0]
        
        dim_th = dims[dim_mapping['th']][0][0]
        dim_lam= dims[dim_mapping['lam']][0][0]
        dim_neq = dims[dim_mapping['neq']][0][0]        
                
        inds['deltaTheta'].append(np.arange(shift, dim_th + shift))
        shift = shift + dim_th
        inds['deltaLambda'].append(np.arange(shift, dim_lam + shift))
        shift = shift + dim_lam
        inds['deltaNu'].append(np.arange(shift, dim_neq + shift))
        shift += dim_neq


        inds['rDual'].append(np.arange(shift2, dim_th + shift2))
        shift2 += dim_th
        inds['rCent'].append(np.arange(shift2, dim_lam + shift2))
        shift2 += dim_lam
        inds['rPri'].append(np.arange(shift2, dim_neq + shift2))
        shift2 += dim_neq
    
    

    general_mapping= make_mapping(model[0][0][0][0][0])
    dims_general = model[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)    
    p = dims_general[0][0][dims_general_mapping['p']][0][0]
    inds['deltaP'] = np.arange(shift, p+shift)
    shift += p
    inds['deltaLambdaP'] = np.arange(shift, shift + 2*p)
    shift += 2 * p
    inds['rdualP'] = np.arange(shift2, shift2 + p)
    shift2 += p
    inds['rcentP'] = np.arange(shift2, shift2 + 2*p)
    shift2 += 2 * p
    
    new_column_inds = np.zeros(shift, dtype = 'int')
    shift = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_lam= dims[dim_mapping['lam']][0][0]
        new_column_inds[shift:shift + dim_lam] = inds['deltaLambda'][k]
        shift += dim_lam
    
    new_column_inds[shift: 2 * p + shift] = inds['deltaLambdaP']
    shift += 2 * p
    B0_nrows = shift # All the lambda changes
    
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_th = dims[dim_mapping['th']][0][0]
        new_column_inds[shift:dim_th + shift] = inds['deltaTheta'][k]
        shift += dim_th
    
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_neq = dims[dim_mapping['neq']][0][0]          
        new_column_inds[shift:dim_neq+shift] = inds['deltaNu'][k]
        shift += dim_neq
    
    new_column_inds[shift:p+shift] = inds['deltaP']
    shift += p
    
    new_rows_inds = np.zeros(shift2, dtype = 'int')
    
    shift2 = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_lam= dims[dim_mapping['lam']][0][0]
        new_rows_inds[shift2:dim_lam + shift2] = inds['rCent'][k]
        shift2 += dim_lam
    
    new_rows_inds[shift2:2*p + shift2] = inds['rcentP']
    shift2 += 2 * p
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_th= dims[dim_mapping['th']][0][0]
        new_rows_inds[shift2:dim_th + shift2] = inds['rDual'][k]
        shift2 += dim_th
    new_rows_inds[shift2:p + shift2] = inds['rdualP']
    shift2 += p
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_neq= dims[dim_mapping['neq']][0][0]
        new_rows_inds[shift2:dim_neq + shift2] = inds['rPri'][k]
        shift2 += dim_neq
    
    return([new_rows_inds, new_column_inds], [B0_nrows, dims_general[0][0][dims_general_mapping['th']][0][0]])



def parallel_newton(con_num):


    model_specific = get_model_specific(model, con_num)
    info_mapping= make_mapping(model_specific)
    dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
    nieq = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['nineq']][0][0]   

    # TODO: should use a mapping for A2, D2 etc...
    A2 = model_specific[0][0][8] 
    D2 = model_specific[0][0][9]
    B =  model_specific[0][0][5]    
    m = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['m']][0][0]   

    gval = np.zeros((nieq, 1))
    gval[:m] = D2.dot(A2).dot(theta) - constr_data['constr'][0][0][constr_mapping['uf']][0][con_num]
    gval[m:2*m] = -D2.dot(A2).dot(theta) + constr_data['constr'][0][0][constr_mapping['lf']][0][con_num]
    gval = gval.flatten()

    Dg = np.concatenate((D2.dot(A2),-D2.dot(A2)))
    Ax = B

    dim_th = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['th']][0][0]        
    dim_n = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['n']][0][0]
    

    ab = (y0[0][0][dim_y0_mapping['lam']][0][con_num]).flatten()
    LM = -1 * np.diag(ab)

    LM = LM.dot(Dg)

    r1_M = [np.zeros((dim_th, dim_th)),    Dg.T ,                       Ax.T]
    r2_M = [LM,                 -np.diag(gval),               np.zeros((nieq, dim_n))]
    r3_M = [Ax,                  np.zeros((dim_n, nieq)),     np.zeros((dim_n, dim_n))]
    r_M = [r1_M,r2_M,r3_M]

    r_M = sparse.csr_matrix(np.block(r_M))


    Au = -1 * np.eye(dim_n)

    r1_B = [np.zeros((dim_th, dim_n)), np.zeros((dim_th, lam_u_s))]
    r2_B = [np.zeros((nieq, dim_n)), np.zeros((nieq, lam_u_s))]
    r3_B = [Au, np.zeros((dim_n, lam_u_s))]
    r_B = [r1_B,r2_B,r3_B]

    r_B = sparse.csr_matrix(np.block(r_B))
    
    #return None

    return r_M,r_B



'''====================================================================='''
'''========================CREATE NEWTON MATRIX========================='''
'''====================================================================='''

nbus = sys.argv[1]
cont = sys.argv[2]

model_data = sio.loadmat("Data/model_"+nbus+"_nbus"+str(cont)+"_ncont")
y0_data = sio.loadmat("Data/y0_"+nbus+"_nbus"+str(cont)+"_ncont")
constr_data = sio.loadmat("Data/constr_"+nbus+"_nbus"+str(cont)+"_ncont")
#rhs = sio.loadmat('rhs')['rhs']

model = model_data['model']
y0 = y0_data['y0']

dim_y0_mapping = make_mapping(y0[0]) 
lam_u_s = np.size(y0[0][0][dim_y0_mapping['lam_u']])   
theta = y0[0][0][dim_y0_mapping['th']][0][0]    

constr_mapping = make_mapping(constr_data['constr'][0])

#Total specific to consider
total_cons = len(model[0][0][1][0])


#Create the M and the Bu matri foe each contigency
nproc = int(sys.argv[3])

print('running for total cons: {0} with nproc {1}'.format(cont, nproc))

Ms = []
Bu = []
pool = mp.Pool(processes = nproc)
ans = pool.map(parallel_newton,np.arange(total_cons))

for con in ans:
    Ms.append(con[0])
    Bu.append(con[1])

'''
#if wanting to run serial code...
Ms = []
Bu = []
start_t = time.time()
for con_num in range(total_cons):
    r_M,r_B = parallel_newton(con_num)
    Ms.append(r_M))
    Bu.append(r_B)
t = time.time()-start_t
print('serial time:',t)
'''

g_u = np.zeros((lam_u_s, 1));
    
    
general_mapping= make_mapping(model[0][0][0][0][0])
dims_general = model[0][0][0][0][0][general_mapping['dim']]
dims_general_mapping = make_mapping(dims_general)

#TODO: Fix naming...
n = dims_general[0][0][dims_general_mapping['p']][0][0]
p = y0[0][0][dim_y0_mapping['p']]
g_u[:n] = p - constr_data['constr'][0][0][constr_mapping['up']]
g_u[n:2*n] = -p + constr_data['constr'][0][0][constr_mapping['lp']];


Dg_u = np.concatenate((np.eye(n), -np.eye(n)))
pars_general = model[0][0][0][0][0][general_mapping['par']]
pars_mapping = make_mapping(pars_general)
par_W = pars_general[0][0][pars_mapping['W']]

temp = np.diag(y0[0][0][dim_y0_mapping['lam_u']].flatten())
r1 = [2*par_W, Dg_u.T]
r2 = [-temp.dot(Dg_u), -1 * np.diag(g_u.flatten())]
Du = np.block([r1, r2])

BB_nrows = dims_general[0][0][dims_general_mapping['BB_nrows']][0][0]
BB_ncols = dims_general[0][0][dims_general_mapping['BB_ncols']][0][0]
BB = np.zeros((BB_nrows, BB_ncols))

#This shift is just keep track of where we need to update in the block matrices
shift = 0
for con_num in range(total_cons):
    model_specific_i = get_model_specific(model, con_num)
    specific_mapping = make_mapping(model_specific_i)
    dims = model_specific_i[0][0][specific_mapping['dim']]
    dims_mapping = make_mapping(dims)
    BB_nrows = dims[0][0][dims_mapping['BB_nrows']][0][0]
    BB[shift :BB_nrows + shift, : ] = Bu[con_num].toarray()
    shift += BB_nrows

if total_cons == 0:
    AA = Ms[0]
else:
    AA = sparse.block_diag(Ms)

# Combine it all together! 

'''
    AA: Combination of all the Ms
    Du: Primary/dual variables related to the initial physical construactions
    BB: something in betweenn...(should be a bit more clear)
    '''
M_newton = sparse.bmat([[AA, sparse.csc_matrix(BB)], [sparse.csc_matrix(BB).T, sparse.csc_matrix(Du)]])



#AB = run_sample()
#ABC = run_sample2()
#akc = run_many()

def fwd_back(b, L, U):
    '''SImple forward bck sub'''
    inter = sparse.linalg.spsolve_triangular(L, b)
    x = sparse.linalg.spsolve_triangular(U, inter, False)
    return(x)

'''Handle getting Schur complement from L, U
TODO Actually do it - for now just return'''
def get_S(L, U, num = 33):
    return(L[num:, num:], U[num:, num:])


def get_interface(size):
    return np.zeros((size, 1))


def grimes_solver(Ai, fi, gi, niter, useMPI):
    combined = np.concatenate((fi, gi))
    ILU = sparse.linalg.spilu(Ai)
    (L1, U1) = (ILU.L, ILU.U)
    yi = 0
    r = fwd_back(combined, L1, U1)
    Pr = r[len(fi):]
    V = np.zeros((len(Pr), niter + 1))
    beta = np.linalg.norm(Pr)
    v1 = Pr / beta
    (LS, US) = get_S(L1, U1)
    Hs = np.zeros((niter + 1, niter + 1))
    V[:, 0] = v1.flatten()
    for j in range(0, niter):
        yinterface = get_interface(LS.shape[0])
        t= fwd_back(yinterface, LS, US )
        w = V[:, j] + t.T
        for l in range(0, j):
            Hs[l, j] = w.dot(V[:, l])
            w = w - Hs[l, j] * (V[:, l])
        Hs[j+1, j] = np.linalg.norm(w)
        V[:, j+1]= w/ Hs[j+1, j]
    Hs = Hs[:niter, :niter]
    V = V[:, :niter]
    z = np.linalg.lstsq(Hs, np.ones((niter)) * beta)[0]
    yi = yi + V.dot(z)
    yinterface = get_interface(LS.shape[0]) 
    t = yinterface
    gi = gi - t
    combined = np.concatenate((fi, gi))
    combined_soln = fwd_back(combined, L1, U1)
    return(combined_soln[:len(fi)], combined_soln[len(fi):])
    

#B = A.tocsr()
