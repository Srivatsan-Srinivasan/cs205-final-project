# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:12:31 2019

@author: Aditya
"""

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.io as sio


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
  permuted_row = orderRow[M.row];
  permuted_col = orderCol[M.col];
  new_M = sparse.coo_matrix((M.data, (permuted_row, permuted_col)), shape=M.shape);
  return new_M;

    

def multigrid(A, rhs, current_level, terminal_level,
              pMethod = 'identity', nMethod = 'given', pInput = None, nInput = None ):

 #   assert(PMethod[pMethod])
 #   assert(nrowMethod[nMethod])

    # TODO: Should check if it's part of the enum     
    if(pMethod == 'given'):
        if(len(pInput) != terminal_level - current_level):
            raise ValueError("Needs to be consistent")
        else:
            A = permute_sparse_matrix(A, pInput[0][0], pInput[0][1])
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
        y0 = multigrid(A1, g0_prime, current_level + 1, terminal_level, 'identity', 'given', pInput, nInput)
        u0 = spsolve(U0,(f0_prime - W0 * y0))
        u0 = u0.reshape(len(u0), 1)
        y0 = np.concatenate((u0,y0))
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


def create_M_newton(model_data, y0_data, constr_data):

    #TODO - should have these as arguments....
    model = model_data['model']
    y0 = y0_data['y0']
    
    dim_y0_mapping = make_mapping(y0[0])    
    theta = y0[0][0][dim_y0_mapping['th']][0][0]    
    
    Ms = []
    Bu = []
    #Total specific to consider
    total_cons = len(model[0][0][1][0])
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
    
        constr_mapping = make_mapping(constr_data['constr'][0])
        
        gval = np.zeros((nieq, 1))
        gval[:m] = D2.dot(A2).dot(theta) - constr_data['constr'][0][0][constr_mapping['uf']][0][con_num]
        gval[m:2*m] = -D2.dot(A2).dot(theta) + constr_data['constr'][0][0][constr_mapping['lf']][0][con_num]
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
    g_u[:n] = p - constr_data['constr'][0][0][constr_mapping['up']]
    g_u[n:2*n] = -p + constr_data['constr'][0][0][constr_mapping['lp']];
    
    
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
    
    print(shift)
    new_column_inds = np.zeros(shift, dtype = 'int')
    shift = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_lam= dims[dim_mapping['lam']][0][0]
        print(dim_lam)
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

#AB = run_sample()
ABC = run_sample2()