# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:12:31 2019

@author: Aditya
"""

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.io as sio

def multigrid(A, rhs, current_level, terminal_level, nrows, inds):
    
 #   nrows = get_splits(A)
  #  inds = permuate(A)
    A = A[:, inds]
    A = A[inds, :]
    rhs = rhs[inds]

    # Split input matrix based on nrows    
    #TODO: Actual algorithm for nrwos.
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
    f0_prime = spsolve(L0,f0);
    f0_prime = f0_prime.reshape(len(f0_prime), 1)
    g0_prime = g0 - G0 * f0_prime;
    
    if(current_level == terminal_level):
        # Direct Solve via ILU
        ILU = sparse.linalg.spilu(B0)
        (L1, U1) = (ILU.L, ILU.U)
        G1 = spsolve(U1.T, E0.T)
        W1 = spsolve(L1, F0)
        A2 = C0 - G1.T * W1 
        
        # Backsolve 
        f1_prime = spsolve(L1, f0)
        f1_prime = f1_prime.reshape(len(f1_prime), 1)
        inner = G1.T * f1_prime
 #       inner = inner.reshape((len(inner), 1))
        g1_prime = g0 - inner
        
        #More backsolve
        y1 = spsolve(A2, g1_prime)
        y1 = y1.reshape(len(y1), 1)
        u1 = spsolve(U1,(f1_prime.reshape(len(f1_prime), 1) - W1 * y1))
        u1 = u1.reshape(len(u1), 1)
        y0 = np.concatenate((u1,y1))
        #Stick them together
        return y0
    else:
        # Descent donwards
        
        #Todo: Shouldn't hardcode 9
        y0 = multigrid(A1, g0_prime, current_level + 1, terminal_level, 8, np.arange(len(g0_prime)))
        u0 = spsolve(U0,(f0_prime - W0 * y0))
        u0 = u0.reshape(len(u0), 1)
        y0 = np.concatenate((u0,y0))
        return y0


def run_sample():
    data = sio.loadmat("M_newton")
    A = data["M_newton"]
    data = sio.loadmat("rhs")
    rhs = data["rhs"]

    res = multigrid(A, rhs, 1, 2, 32, np.arange(len(rhs)))
    return(res)



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


def create_M_newton():

    #TODO - should have these as arguments....
    data2 = sio.loadmat("5_bus_model")
    model = data2['model']
    data2 = sio.loadmat("5_bus_model_y0")
    data3 = sio.loadmat("5_bus_model_const")
    y0 = data2['y0']
    
    dim_y0_mapping = make_mapping(data2['y0'][0])    
    theta = data2['y0'][0][0][dim_y0_mapping['th']][0][0]    
    
    Ms = []
    Bu = []
    #Total contingencies in this set 
    total_cons = 2
    #Create the M and the Bu matri foe each contigency
    for con_num in range(total_cons):
        model_specific = get_model_specific(model, con_num)
        info_mapping= make_mapping(model_specific)
        dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
        nieq = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['nineq']][0][0]   
        
        # TODO: should use a mapping for A2, D2 etc...
        A2 = model_specific[0][0][8] 
        D2 = model_specific[0][0][9]
        B =  model_specific[0][0][5]    
        m = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['m']][0][0]   
    
        constr_mapping = make_mapping(data3['constr'][0])
        
        gval = np.zeros((nieq, 1))
        gval[:m] = D2.dot(A2).dot(theta) - data3['constr'][0][0][constr_mapping['uf']][0][con_num]
        gval[m:2*m] = -D2.dot(A2).dot(theta) + data3['constr'][0][0][constr_mapping['lf']][0][con_num]
        gval = gval.flatten()
        
        Dg = np.concatenate((D2.dot(A2),-D2.dot(A2)))
        Ax = B
        
        dim_th = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['th']][0][0]        
        dim_n = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['n']][0][0]
            
        LM = -np.diag((y0[0][0][dim_y0_mapping['lam']][0][con_num]).flatten())
        LM = LM.dot(Dg)
        
        r1 = [np.zeros((dim_th, dim_th)),    Dg.T ,                       Ax.T]
        r2 = [LM,                 -np.diag(gval),               np.zeros((nieq, dim_n))]
        r3 = [Ax,                  np.zeros((dim_n, nieq)),     np.zeros((dim_n, dim_n))]
        
        Ms.append(sparse.csc_matrix(np.block([r1, r2, r3])))
                
        Au = - np.eye(dim_n)
        lam_u_s = np.size(y0[0][0][dim_y0_mapping['lam_u']])
        r1 = [np.zeros((dim_th, dim_n)), np.zeros((dim_th, lam_u_s))]
        r2 = [np.zeros((nieq, dim_n)), np.zeros((nieq, lam_u_s))]
        r3 = [-Au, np.zeros((dim_n, lam_u_s))]
        Bu.append(sparse.csc_matrix(np.block([r1, r2, r3])))
    
    
    #Constructing the general matrix - need to know the variables
    g_u = np.zeros((lam_u_s, 1));
    
    
    general_mapping= make_mapping(model[0][0][0][0][0])
    dims_general = model[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)
    
    #TODO: Fix naming...
    n = dims_general[0][0][dims_general_mapping['p']][0][0]
    p = y0[0][0][dim_y0_mapping['p']]
    g_u[:n] = p - data3['constr'][0][0][constr_mapping['up']]
    g_u[n:2*n] = -p + data3['constr'][0][0][constr_mapping['lp']];
    
    
    Dg_u = np.concatenate((np.eye(n), -np.eye(n)))
    pars_general = model[0][0][0][0][0][general_mapping['par']]
    pars_mapping = make_mapping(pars_general)
    par_W = pars_general[0][0][pars_mapping['W']]
    
    temp = np.diag(y0[0][0][dim_y0_mapping['lam_u']].flatten())
    r1 = [2*par_W, Dg_u.T]
    r2 = [-temp.dot(Dg_u), -np.diag(g_u.flatten())]
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
    return(M_newton)