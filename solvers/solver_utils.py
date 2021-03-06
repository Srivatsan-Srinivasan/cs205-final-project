'''
SOLVER SPECIFIC FUNCTIONS

Solver Specific Utils for Newton Step Solvers in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
import os
import sys
import scipy.io as sio
from mpi4py import MPI

def make_mapping(d):
    '''
    Given a model - like data structure, constructs the equivalent mapping dict.
    
    Args:
        d (any model - like): model - like data structure to be converted

    Returns:
        mapping (dict): dictionary definiting the equivalent mapping
    '''

    key = d.dtype.fields.keys()
    val = np.arange(len(key))

    mapping = {k:v for k, v in zip(key, val)}
    return mapping

def get_model_specific(model_data, constr_num):    
    '''
    Given a loaded model data file matrix, returns the constraints specific model information.
    
    Args:
        model_data (scipy.sparse or numpy matrices): model_data
        constr_num (int): index for the constraint of interest

    Returns:
        constr_info (scipy.sparse or numpy matrices): constraint specific model information
    '''


    '''Get the constraint specific model information'''
    return model_data[0][0][1][0][constr_num]

def construct_NewtonBDMatrix_PARALLEL_worker(con_num):
    '''
    Given a contingency number from construct_NewtonBDMatrix fn., returns the corresponding r_M, r_B deconstruction per report/website.

    Args:
        con_num (int): the index of the contingency being looked at

    Returns:
        r_M (scipy.sparse matrix): the sparse matrix M that is deconstructed from A
        r_B (scipy.sparse matrix): the sparse matrix B that is deconstructed from B
    '''
    #set local variables to global variables
    model = MODEL
    lam_u_s = LAM_U_S
    y0 = Y0
    dim_y0_mapping = DIM_Y0_MAPPING
    theta = THETA
    constr_mapping = CONSTR_MAPPING
    constr_data = CONSTR_DATA
    #set local variables to global variables

    model_specific = get_model_specific(model, con_num)
    info_mapping= make_mapping(model_specific)
    dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
    nieq = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['nineq']][0][0]   
    
    A2 = model_specific[0][0][8] 
    D2 = model_specific[0][0][9]
    B =  model_specific[0][0][5]    
    m = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['m']][0][0]   

    gval = np.zeros((nieq, 1))
    gval[:m] = D2.dot(A2).dot(theta) - constr_data[0][0][constr_mapping['uf']][0][con_num]
    gval[m:2*m] = -D2.dot(A2).dot(theta) + constr_data[0][0][constr_mapping['lf']][0][con_num]
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

    return r_M,r_B

def construct_NewtonBDMatrix_PARALLEL(model_data, y0_data, constr_data, nproc=0):
    '''
    Given the model, Y0 variables (per report/website), constraint data file matrix, number of processors
    to use in multiprocessing, constructs the newton block diagonal matrix per report/website using pyMP

    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        y0_data (scipy.sparse or numpy matrices): Y0 variables data file matrix per report/website
        constr_data (scipy.sparse or numpy matrices): constraint data file matrix
        nproc (int): number of processors to use in multiprocessing

    Returns:
        newton_matrix (scipy.sparse or numpy matrices): newton block diagonal matrix per report/website
    '''

    #set global variables to local variables
    model = model_data
    y0 = y0_data
    global MODEL 
    MODEL = model    
    global Y0 
    Y0 = y0

    dim_y0_mapping = make_mapping(y0[0])
    global DIM_Y0_MAPPING 
    DIM_Y0_MAPPING = dim_y0_mapping 
    lam_u_s = np.size(y0[0][0][dim_y0_mapping['lam_u']])
    global LAM_U_S 
    LAM_U_S = lam_u_s   
    theta = y0[0][0][dim_y0_mapping['th']][0][0]
    global THETA 
    THETA = theta    

    constr_mapping = make_mapping(constr_data[0])
    global CONSTR_MAPPING 
    CONSTR_MAPPING = constr_mapping

    global CONSTR_DATA
    CONSTR_DATA = constr_data
    #set global variables to local variables

    #Total specific to consider
    total_cons = len(model[0][0][1][0])

    #Create the M and the Bu matri foe each contigency

    if nproc == 0:
        nproc = total_cons

    Ms = []
    Bu = []
    pool = mp.Pool(processes = nproc)
    ans = pool.map(construct_NewtonBDMatrix_PARALLEL_worker,np.arange(total_cons))

    for con in ans:
        Ms.append(con[0])
        Bu.append(con[1])

    del MODEL, Y0, DIM_Y0_MAPPING, LAM_U_S, THETA, CONSTR_MAPPING

    g_u = np.zeros((lam_u_s, 1));
                
    general_mapping= make_mapping(model[0][0][0][0][0])
    dims_general = model[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)
    
    n = dims_general[0][0][dims_general_mapping['p']][0][0]
    p = y0[0][0][dim_y0_mapping['p']]
    g_u[:n] = p - constr_data[0][0][constr_mapping['up']]
    g_u[n:2*n] = -p + constr_data[0][0][constr_mapping['lp']];

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

    #shift to track update locations in block matrices
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

    #combine it all together!
    #AA: Combination of all the Ms
    #Du: Primary/dual variables related to the initial physical constructions
    #BB: All the stuff in between
    newton_matrix = sparse.bmat([[AA, sparse.csc_matrix(BB)], [sparse.csc_matrix(BB).T, sparse.csc_matrix(Du)]])

    return newton_matrix


def construct_NewtonBDMatrix(model_data, y0_data, constr_data):
    '''
    Given the model, Y0 variables (per report/website), constraint data file matrix, constructs
    the newton block diagonal matrix per report/website.
    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        y0_data (scipy.sparse or numpy matrices): Y0 variables data file matrix per report/website
        constr_data (scipy.sparse or numpy matrices): constraint data file matrix
    Returns:
        newton_matrix (scipy.sparse or numpy matrices): newton block diagonal matrix per report/website
    '''

    dim_y0_mapping = make_mapping(y0_data[0])
    theta = y0_data[0][0][dim_y0_mapping['th']][0][0]

    Ms = []
    Bu = []
    #total specific cons to consider
    total_cons = len(model_data[0][0][1][0])
    #construct M and B for each contingency
    for con_num in range(total_cons):
        model_specific = get_model_specific(model_data, con_num)
        info_mapping = make_mapping(model_specific)
        dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
        nieq = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['nineq']][0][0]

        A2 = model_specific[0][0][8]
        D2 = model_specific[0][0][9]
        B = model_specific[0][0][5]
        m = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['m']][0][0]

        constr_mapping = make_mapping(constr_data[0])

        gval = np.zeros((nieq, 1))
        gval[:m] = D2.dot(A2).dot(theta) - constr_data[0][0][constr_mapping['uf']][0][con_num]
        gval[m:2*m] = -D2.dot(A2).dot(theta) + \
            constr_data[0][0][constr_mapping['lf']][0][con_num]
        gval = gval.flatten()

        Dg = np.concatenate((D2.dot(A2), -D2.dot(A2)))
        Ax = B

        dim_th = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['th']][0][0]
        dim_n = model_specific[0][0][info_mapping['dim']][0][0][dim_mapping['n']][0][0]

        ab = (y0_data[0][0][dim_y0_mapping['lam']][0][con_num]).flatten()
        LM = -1 * np.diag(ab)
        LM = LM.dot(Dg)

        r1 = [np.zeros((dim_th, dim_th)),          Dg.T,                       Ax.T]
        r2 = [LM,                        -np.diag(gval),    np.zeros((nieq, dim_n))]
        r3 = [Ax,               np.zeros((dim_n, nieq)),   np.zeros((dim_n, dim_n))]

        Ms.append(sparse.csr_matrix(np.block([r1, r2, r3])))

        Au = -1 * np.eye(dim_n)
        lam_u_s = np.size(y0_data[0][0][dim_y0_mapping['lam_u']])
        r1 = [np.zeros((dim_th, dim_n)), np.zeros((dim_th, lam_u_s))]
        r2 = [np.zeros((nieq, dim_n)),   np.zeros((nieq, lam_u_s))]
        r3 = [Au,                        np.zeros((dim_n, lam_u_s))]
        Bu.append(sparse.csr_matrix(np.block([r1, r2, r3])))

    #constructing the general matrix - need to know the variables
    g_u = np.zeros((lam_u_s, 1))

    general_mapping = make_mapping(model_data[0][0][0][0][0])
    dims_general = model_data[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)
    
    n = dims_general[0][0][dims_general_mapping['p']][0][0]
    p = y0_data[0][0][dim_y0_mapping['p']]
    g_u[:n] = p - constr_data[0][0][constr_mapping['up']]
    g_u[n:2*n] = -p + constr_data[0][0][constr_mapping['lp']]

    Dg_u = np.concatenate((np.eye(n), -np.eye(n)))
    pars_general = model_data[0][0][0][0][0][general_mapping['par']]
    pars_mapping = make_mapping(pars_general)
    par_W = pars_general[0][0][pars_mapping['W']]

    temp = np.diag(y0_data[0][0][dim_y0_mapping['lam_u']].flatten())
    r1 = [2*par_W, Dg_u.T]
    r2 = [-temp.dot(Dg_u), -1 * np.diag(g_u.flatten())]
    Du = np.block([r1, r2])

    BB_nrows = dims_general[0][0][dims_general_mapping['BB_nrows']][0][0]
    BB_ncols = dims_general[0][0][dims_general_mapping['BB_ncols']][0][0]
    BB = np.zeros((BB_nrows, BB_ncols))

    #shift to track update locations in block matrices
    shift = 0
    for con_num in range(total_cons):
        model_specific_i = get_model_specific(model_data, con_num)
        specific_mapping = make_mapping(model_specific_i)
        dims = model_specific_i[0][0][specific_mapping['dim']]
        dims_mapping = make_mapping(dims)
        BB_nrows = dims[0][0][dims_mapping['BB_nrows']][0][0]
        BB[shift:BB_nrows + shift, :] = Bu[con_num].toarray()
        shift += BB_nrows

    if total_cons == 0:
        AA = Ms[0]
    else:
        AA = sparse.block_diag(Ms)

    #combine it all together!
    #AA: Combination of all the Ms
    #Du: Primary/dual variables related to the initial physical constructions
    #BB: All the stuff in between

    newton_matrix = sparse.bmat([
        [AA, sparse.csc_matrix(BB)], 
        [sparse.csc_matrix(BB).T, sparse.csc_matrix(Du)]])

    return(newton_matrix)

def get_specific_dim(model_data, cons_num):    
    '''
    Given the model data and the constraint number, returns specific information about
    the dim per report/website.
    
    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        cons_num (int): constraint number

    Returns:
        dims (int): number of dims
        dim_mapping (dict): equivalent dict mapping about the requested specific information
    '''
    model_specific = get_model_specific(model_data, cons_num)
    info_mapping= make_mapping(model_specific)
    dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
    dims = model_specific[0][0][info_mapping['dim']][0][0]
    return((dims, dim_mapping))

def get_num_cont(model_data):
    '''
    Given the model data returns the number of contengencies.    
    
    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix    

    Returns:
        num_cont (int): number of contengencies        
    '''

    return(len(model_data[0][0][1][0]) - 1)

def permute_NewtonBDMatrix(model_data, method = "standard"):
    '''
    Given the model data, and a permutation method to use,
    intelligently permutes the newton matrix to optimize runtime.
    
    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        method (str): string defining a valid permutation method to employ

    Returns:
        (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        (list of numpy arrays): new orderings defining shift to track updates (???)

    available method types:
    standard: (group lambdas together and then group (nu/theta))
    grouped: (group all lambdas together than all nu and then all theta)
    '''

    #The algorithm here is to permute to get the same type of variables together. 
    #Roughtly speaking instead of grouping by contingency directly we want to group by 
    #theta variables, lambda variables etc. This somehow gives us a very diagonal 
    #B0 matrix which is really easy to work with. 
    
    if(method not in ['standard', 'grouped']):
        raise ValueError("Invalid method supplied")
    # method types: 
    # standard (group lambdas together and then group (nu/theta))
    # grouped (group all lambdas together than all nu and then all theta)
    ncont = get_num_cont(model_data)
    inds = {'deltaTheta': [], 'deltaLambda': [], 'deltaNu': [],
            'rDual': [], 'rCent': [], 'rPri': []}

    shift = 0
    shift2 = 0
    for cons_num in range(0, ncont + 1):
        model_specific = get_model_specific(model_data, cons_num)
        info_mapping = make_mapping(model_specific)
        dim_mapping = make_mapping(
            model_specific[0][0][info_mapping['dim']][0])
        dims = model_specific[0][0][info_mapping['dim']][0][0]

        dim_th = dims[dim_mapping['th']][0][0]
        dim_lam = dims[dim_mapping['lam']][0][0]
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

    general_mapping = make_mapping(model_data[0][0][0][0][0])
    dims_general = model_data[0][0][0][0][0][general_mapping['dim']]
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

    new_column_inds = np.zeros(shift, dtype='int')
    shift = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model_data, k)
        dim_lam = dims[dim_mapping['lam']][0][0]
        new_column_inds[shift:shift + dim_lam] = inds['deltaLambda'][k]
        shift += dim_lam

    new_column_inds[shift: 2 * p + shift] = inds['deltaLambdaP']
    shift += 2 * p
    B0_nrows = shift  # all the lambda changes

    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model_data, k)
        dim_th = dims[dim_mapping['th']][0][0]
        new_column_inds[shift:dim_th + shift] = inds['deltaTheta'][k]
        shift += dim_th

        if(method == 'grouped'):
            dim_neq = dims[dim_mapping['neq']][0][0]
            new_column_inds[shift:dim_neq+shift] = inds['deltaNu'][k]
            shift += dim_neq
        
       
    if(method == "standard"):
        for k in range(0, ncont + 1):
            (dims, dim_mapping) = get_specific_dim(model_data, k)
            dim_neq = dims[dim_mapping['neq']][0][0]          
            new_column_inds[shift:dim_neq+shift] = inds['deltaNu'][k]
            shift += dim_neq
       
    new_column_inds[shift:p+shift] = inds['deltaP']
    shift += p

    new_rows_inds = np.zeros(shift2, dtype='int')

    shift2 = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model_data, k)
        dim_lam = dims[dim_mapping['lam']][0][0]
        new_rows_inds[shift2:dim_lam + shift2] = inds['rCent'][k]
        shift2 += dim_lam

    new_rows_inds[shift2:2*p + shift2] = inds['rcentP']
    shift2 += 2 * p
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model_data, k)
        dim_th = dims[dim_mapping['th']][0][0]
        new_rows_inds[shift2:dim_th + shift2] = inds['rDual'][k]
        shift2 += dim_th

        if(method == "grouped"):
            dim_neq = dims[dim_mapping['neq']][0][0]
            new_rows_inds[shift2:dim_neq + shift2] = inds['rPri'][k]
            shift2 += dim_neq

    new_rows_inds[shift2:p + shift2] = inds['rdualP']
    shift2 += p
    if(method == 'standard'):
        for k in range(0, ncont + 1):
            (dims, dim_mapping) = get_specific_dim(model_data, k)
            dim_neq= dims[dim_mapping['neq']][0][0]
            new_rows_inds[shift2:dim_neq + shift2] = inds['rPri'][k]
            shift2 += dim_neq

    return [new_rows_inds, new_column_inds, inds], [B0_nrows, dims_general[0][0][dims_general_mapping['th']][0][0]]

def permute_sparse_matrix(M, orderRow, orderCol):    
    ''' 
    Permutes a sparse matrix given a custom ordering of the row and column indices.
    Taken from https://gist.github.com/vtraag/8b82e10e57d93eacc524

    Args:
        M (scipy.sparse matrices): scipy sparse matrice to permute
        orderRow (list of ints): custom ordering of row indices
        orderCol (list of ints): custom ordering of column indices

    Returns:
        M_permuted (scipy.sparse matrices): permuted scipy sparce matrice
    '''

    M_permuted = M.tocsr()
    M_permuted = M_permuted[:, orderCol]
    M_permuted = M_permuted[orderRow, :]
    return M_permuted    

def repermute(y0, new_col_inds):    
    ''' 
    Permutes a sparse matrix given a custom ordering of the row and column indices.
    Taken from https://gist.github.com/vtraag/8b82e10e57d93eacc524

    Args:
        M (scipy.sparse matrices): scipy sparse matrice to permute
        orderRow (list of ints): custom ordering of row indices
        orderCol (list of ints): custom ordering of column indices

    Returns:
        M_permuted (scipy.sparse matrices): permuted scipy sparce matrice
    '''
    y0_reorder = np.zeros_like(y0)
    for i in range(0, len(y0)):
        itemindex = np.where(new_col_inds == i)[0]
        y0_reorder[i] = y0[itemindex]
    return(y0_reorder)

def multigrid(A, rhs, current_level, terminal_level,
              pMethod = 'identity', nMethod = 'given', pInput = None, nInput = None ):
    ''' 
    Implemented the multigrid procedure per report/website.

    Args:
        A (scipy.sparse or numpy matrices): Newton step matrix to be solved
        rhs (scipy.sparse or numpy matrices): RHS variables per report/website
        current_level (int): current descent level in the multigrid procedure
        terminal_level (int): terminal descent level in the multigrid procedure
        pMethod (string): valid permutation method for newton step matrix
        nMethod (string): valid method for dealing with nInput
        pInput (scipy.sparse or numpy matrices): pInput to handle per pMethod
        nInput (scipy.sparse or numpy matrices): nInput to handle per nMethod

    Returns:
        y0 (scipy.sparse matrix): solution to the newton step problem per report/website        
    '''
    if(pMethod == 'given'):
        if(len(pInput) != terminal_level - current_level):
            raise ValueError("Needs to be consistent")
        else:        
            A = permute_sparse_matrix(A, pInput[0][0], pInput[0][1])            
            rhs = rhs[pInput[0][0]]
            pMethod = 'identity'
        
        
    if(nMethod == 'given'):
        if(len(nInput) != terminal_level - current_level + 1):
            raise ValueError("Needs to be consistent")
        else:
            nrows = nInput[0]
            if(terminal_level != current_level):
                nInput = nInput[1:]
    
    # Split input matrix based on nrows    
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
    if(current_level == terminal_level):
        # Direct Solve via ILU        
        ILU = sparse.linalg.spilu(B0 + 1e-4 * sparse.eye(B0.shape[0]))
        (L1, U1) = (ILU.L, ILU.U)
        G1 = sparse.linalg.spsolve_triangular(U1.T, (E0.T).todense())
        W1 = sparse.linalg.spsolve_triangular(L1, F0.todense())
        A2 = C0 - G1.T * W1 
        
        # Backsolve 
        f1_prime = sparse.linalg.spsolve_triangular(L1, f0)
        f1_prime = f1_prime.reshape(len(f1_prime), 1)
        inner = G1.T * f1_prime
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
            y0 = repermute(y0, pInput[0][1])
        return y0

def split_newton_matrix(A, rhs, nrows, ncont, nprocess, model, inds):
    ''' 
    When given the partition width for the matrix and various pieces of information about
    the newton step matrix, partitions it.

    Args:
        A (scipy.sparse or numpy matrices): Newton step matrix to be solved
        rhs (scipy.sparse or numpy matrices): RHS variables per report/website
        nrows (int): partition width for the ordered newton step matrix
        ncont (int): number of contengencies
        nprocess (int): number of processors
        model (scipy.sparse or numpy matrices): model data file matrix
        inds (list or numpy array): ordering of indices in newton step matrix

    Returns:
        to_send (nested list or numpy array of scipy.sparse matrices): partitions segements of the initial A matrix to send
        As (list of scipy.sparse matrices): corresponding reconstructed sub-A matrices from partition segments to send
    '''
    B0 = A[:nrows, :nrows]
    F0 = A[:nrows, nrows:]
    E0 = A[nrows:, :nrows]
    C0 = A[nrows:, nrows:]

    f0 = rhs[:nrows]
    g0 = rhs[nrows:]
    (Bs, Fs, Es, Cs, gs, fs, us) = ([], [], [], [], [], [], [])
    
    (start_b, start_f_col, start_f_row, start_e_col, start_e_row,
     start_c_col, start_c_row) = (0, 0, 0, 0, 0, 0, 0)
    for k in range(0, ncont + 1):
        l_cut = len(inds['deltaLambda'][k])
        Bs.append(B0[start_b:start_b + l_cut, start_b:start_b + l_cut])
        us.append(f0[start_b:start_b + l_cut])
        start_b += l_cut
        th_cut = (len(inds['deltaTheta'][k]))

        Fs.append(F0[start_f_row:start_f_row + l_cut, :])
        fs.append(f0[start_f_row:start_f_row + l_cut])

        start_f_row += l_cut
        start_f_col += th_cut
        Es.append(E0[:, start_e_col:start_e_col + l_cut])

        start_e_row += l_cut
        start_e_col += l_cut
        temp_C0 = C0[start_c_row:start_c_row+th_cut, :]
        (r, c) = temp_C0.shape
        CO_above = sparse.csr_matrix((start_c_row, c))
        CO_below = sparse.csr_matrix((c - r - start_c_row, c))
        CO_all = sparse.vstack([CO_above, temp_C0, CO_below, ])
        Cs.append(CO_all)

        temp_g = g0[start_c_row:start_c_row+th_cut]
        g0_above = sparse.csr_matrix((start_c_row, 1))
        g0_below = sparse.csr_matrix((c - r - start_c_row, 1))
        g0_all = sparse.vstack(
            [g0_above, sparse.csc_matrix(temp_g), g0_below, ])
        gs.append(g0_all)

        start_c_row += th_cut

    temp_C0 = C0[start_c_row:, :]
    (r, c) = temp_C0.shape
    CO_above = sparse.csr_matrix((start_c_row, c))
    CO_below = sparse.csr_matrix((c - r - start_c_row, c))
    CO_all = sparse.vstack([CO_above, temp_C0, CO_below])
    Cs.append(CO_all)

    temp_g = g0[start_c_row:]
    g0_above = sparse.csr_matrix((start_c_row, 1))
    g0_below = sparse.csr_matrix((c - r - start_c_row, 1))
    g0_all = sparse.vstack([g0_above, sparse.csc_matrix(temp_g), g0_below, ])
    gs.append(g0_all)

    l_cut_p = len(inds['deltaLambdaP'])
    Bs.append(B0[start_b:start_b + l_cut_p, start_b:start_b + l_cut_p])
    us.append(f0[start_b:start_b + l_cut_p])

    Fs.append(F0[start_f_row:start_f_row + l_cut, :])
    fs.append(f0[start_f_row:start_f_row + l_cut])
    Es.append(E0[:, start_e_col:])
    to_send = [(B, F, E, C, f, g, u)
               for (B, F, E, C, f, g, u) in zip(Bs, Fs, Es, Cs, fs, gs, us)]

    As = []
    for i in range(0, len(Fs)):
        b = Bs[i]
        e = Es[i]
        f = fs[i]
        g = gs[i]
        temp_res = g - e * sparse.diags(1/b.diagonal()) * f
        As.append(temp_res)
    return(to_send, As)

def multigrid_PARALLEL(iproc, combined, inds, nrows, split_solve=None, res=None):
    ''' 
    Implemented a partially parallelized version of the multigrid procedure per report/website.

    Args:
        iproc (int): MPI processor id
        combined (scipy.sparse or numpy matrices): combined local schurs components when descending level
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates (???)
        split_solve (None): unused variable in this partially parallelized implementation
        res (None): unused variable in this partially parallelized implementaiton

    Returns:
        y0 (scipy.sparse matrix): solution to the newton step problem per report/website        
    '''
    newA= combined[0][0].copy()
    newG = combined[0][1].copy()
    for i in range(1, len(combined)):
        newA += combined[i][0]
        newG += combined[i][1]
    otherInfo = [(x[3], x[4], x[5]) for x in combined]    

    cut = nrows[1]
    B0 = newA[:cut, :cut]
    F0 = newA[:cut, cut:]
    E0 = newA[cut:, :cut]
    C0 = newA[cut:, cut:]

    # same for f0. f0 is the stuff that can be computed independently,
    # g0 is the stuff that can't be
    f0 = newG[:cut]
    g0 = newG[cut:]

    
    # Direct Solve via ILU
    
    ILU = sparse.linalg.spilu(B0 + 1e-4 * sparse.eye(B0.shape[0]))
    (L1, U1) = (ILU.L, ILU.U)
    #Finishing ILU
    
    G1 = sparse.linalg.spsolve_triangular(U1.T, (E0.T).todense())
    W1 = sparse.linalg.spsolve_triangular(L1, F0.todense())
    A2 = C0 - G1.T * W1
    #Finished A2
    
    #Backsolve
    f1_prime = sparse.linalg.spsolve_triangular(L1, f0)
    f1_prime = f1_prime.reshape(len(f1_prime), 1)
    inner = G1.T * f1_prime
    g1_prime = g0 - inner
    #Finishing g1_prime
    # More backsolve
    y1 = spsolve(A2, g1_prime)
    #Finished direct solve
    y1 = y1.reshape(len(y1), 1)
    u1 = sparse.linalg.spsolve_triangular(
    U1, (f1_prime.reshape(len(f1_prime), 1) - W1 * y1), False)
    #Finished backsub
    u1 = u1.reshape(len(u1), 1)
    y1 = np.concatenate((u1, y1))
    
    B = sparse.block_diag([combined[i][3] for i in range(0, len(combined)) ])
    F = sparse.vstack([combined[i][4] for i in range(0, len(combined)) ])
    f = np.vstack([combined[i][5] for i in range(0, len(combined)) ])
    #finished constructing
    u0 = sparse.linalg.spsolve_triangular(B, f  - F * y1 , False )
    #finishing backsolve
    u0 = u0.reshape(len(u0), 1)
    y0 = np.concatenate((u0, y1))
    res = repermute(y0, inds[1])

    return res


def local_schurs(inputs):
    ''' 
    Implements the worker logic for distributed calculation of the local schurs components 
    per report/webiste.

    Args:
        inputs (scipy.sparse or numpy matrix): worker split for calculation of the local schurs

    Returns:
        (list of scipy.sparse or numpy matrices): result of local schurs components in workers        
    '''
    (B, F, E, C, f, g, u) = inputs
    A1_local = C - E * sparse.diags(1/B.diagonal()) * F
    g1_local = g - E * sparse.diags(1/B.diagonal()) * f

    return((A1_local, g1_local, u, B, F, f))
    
    
    
def multigrid_full_parallel(iproc, combined, inds, nrows, model, split_solve=None, res=None):
    ''' 
    Implemented a fully parallelized version of the multigrid procedure per report/website.

    Args:
        iproc (int): MPI processor id
        combined (scipy.sparse or numpy matrices): combined local schurs components when descending level
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates
        model (scipy.sparse or numpy matrices): loaded model data file matrix
        split_solve (None): initialization of the iterable for distributed calculations
        res (None): initialization of the solution to the newton step problem per report/website

    Returns:
        res (scipy.sparse matrix): solution to the newton step problem per report/website
    '''
    if(iproc == 0):
        ncont = get_num_cont(model)
        newA= combined[0][0].copy()
        newG = combined[0][1].copy()
        for i in range(1, len(combined)):
            newA += combined[i][0]
            newG += combined[i][1]
        otherInfo = [(x[3], x[4], x[5]) for x in combined]
        split_solve = distributed_data_calc(model, newA, newG, ncont, otherInfo, moveZero = True)
    
    local_inputs = MPI.COMM_WORLD.scatter(split_solve, root=0)
    (B, F, f) = local_inputs[-1]
     
    inner_soln = outer_solver_wrapper(iproc, local_inputs[0], local_inputs[1][0], local_inputs[1][1], 
                             local_inputs[2], B, F, f, (iproc == 0), niter = 10, num_restarts = 10, 
                             tol = 1e-6)       
        
    combined2 = MPI.COMM_WORLD.gather(inner_soln, root = 0)
    if(iproc == 0):
        combined2 = combined2[1:] + [combined2[0]]
        us =  [u[0] for u in combined2]
        ys = [y[1] for y in combined2]
        us = np.concatenate(us)
        ys = np.concatenate(ys)
        res = np.concatenate((us, ys))
        res = repermute(res, inds[1])
        return(res)

def distributed_data_calc(model, A, rhs, num_cont, otherData = None, moveZero= True):
    ''' 
    Implemented the distributed data calc at the terminal level per report/website.    

    Args:
        model (scipy.sparse or numpy matrices): loaded model data file matrix
        A (scipy.sparse or numpy matrix): newton step matrix to be solved
        rhs (scipy.sparse or numpy matrix): RHS variables per report/website
        num_cont (int): number of contengencies
        otherData (list of scipy.sparse or numpy matrices): other data to be 
        passed through the calc
        moveZero (boolean): insert a 0 into the other data to be passed through
        the calc

    Returns:
        grouped (list of tuples of scipy.sparse or numpy matrices): grouped matrices
        of results from split_to_distributed.
    '''
    (Bs, rhs, Hps) = split_to_distributed(model, A, rhs, num_cont)
    if(otherData is not None):
        if(moveZero == True):
            otherData.insert(0, otherData.pop())
        grouped = [(x, y, z, misc) for x, y, z,misc in zip(Bs, rhs, Hps, otherData)]
    else:
        grouped = [(x, y, z) for x, y, z in zip(Bs, rhs, Hps)]
    return(grouped)

def split_to_distributed(model, Ar, rhsr, num_cont):
    ''' 
    Takes model data and a reduced matrix, reorders it and then comes up with a split.

    Args:
        model (scipy.sparse or numpy matrices): loaded model data file matrix
        Ar (scipy.sparse or numpy matrix): reduced matrix of the newton step matrix pre report/website
        rhsr (scipy.sparse or numpy matrix): reduced RHS variables per report/website
        num_cont (int): number of contengencies

    Returns:
        (tuple of scipy.sparse or numpy matrices): tuples of matrices that form the split after reordering
    '''
    Ar2 = Ar
    (cols, rows, inds) = find_regrouping(model)

    to_split_arr = []
    rhs_split_arr = []
    start_r = 0
    start_c = 0
    for i in range(0, num_cont + 1):
        (i_row, i_col) = (len(rows[i]), len(cols[i]))
        to_split_arr.append(Ar2[start_r: start_r+i_row, start_c:start_c+i_col])
        rhs_split = rhsr[start_r:start_r + i_row]
        inter = len(inds['rDual'][i])
        fi = rhs_split[:inter]
        gi = rhs_split[inter:]
        rhs_split_arr.append((fi, gi))
        start_r += i_row
        start_c += i_col
    
    
    check1 = Ar2[:, start_c:]
    check2 = Ar2[start_r:, :]
    assert(sparse.linalg.norm(check1 - check2.T) < 1e-6)
    Cp = Ar2[start_r:, start_c:]
    rhs_left = (np.zeros((0, rhsr.shape[1])), rhsr[start_r:])
    
    to_split_arr.insert(0, Cp)
    rhs_split_arr.insert(0, rhs_left)
    Hps_outer = []
    Hps_inner = []
    start_c = 0
    for i in range(0, num_cont + 1):
        th_len = len(inds['deltaTheta'][i])
        th_nu = len(inds['deltaNu'][i])
        Hps_inner.append(
            Ar2[start_r:, start_c + th_len: start_c + th_len + th_nu])
        Hps_outer.append(
            [Ar2[start_r:, start_c + th_len: start_c + th_len + th_nu]])
    Hps_outer.insert(0, Hps_inner)
    return(to_split_arr, rhs_split_arr, Hps_outer)


def fwd_back(b, L, U):
    '''
    Convenience function implementing the usual forward back substitution. 

    Args:
        b (scipy.sparse or numpy matrix): the matrix to perform the forward backward on
        L (scipy.sparse or numpy matrix): the lower triangular matrix in the forward back
        U (scipy.sparse or numpy matrix): the upper triangular matrix in the forward back

    Returns:
        x (scipy.sparse or numpy matrix): the result of the forward back substitution
    '''
    inter = sparse.linalg.spsolve_triangular(L, b)
    x = sparse.linalg.spsolve_triangular(U, inter, False)
    return(x)


def get_S(L, U, num=33):
    '''
    Convenience function that calculates the schur component from L, U

    Args:
        L (scipy.sparse or numpy matrix): the lower triangular matrix
        U (scipy.sparse or numpy matrix): the upper triangular matrix
        num (int): index valuation defining the partition for L, U matrices

    Returns:
        (tuple of scipy.sparse or numpy matrices): schur component from L, U
    '''
    return(L[num:, num:], U[num:, num:])


def outer_solver_wrapper(iproc, Ai, fi, gi, Hips, B, F, f,  useSchurs, yguess = None, niter = 10, num_restarts = 10, tol = 1e-6):
    '''
    outer wrapper for the main gmres solver

    Args:
        iproc (int): MPI processor int
        Ai (scipy.sparse or numpy matrix): Ai variable for gmres per report/website
        fi (scipy.sparse or numpy matrix): fi variable for gmres per report/website
        gi (scipy.sparse or numpy matrix): gi variable for gmres per report/website
        Hips (scipy.sparse or numpy matrix): Hips variable for gmres per report/website
        B (scipy.sparse or numpy matrix): B variable for gmres per report/website
        F (scipy.sparse or numpy matrix): F variable for gmres per report/website
        f (scipy.sparse or numpy matrix): f variable for gmres per report/website
        useSchurs (boolean): boolean to use or not to use schurs in gmres solver
        yguess (scipy.sparse or numpy matrix): the current best guess for y matrix
        niter (int): number of iterations to use for the main gmres solver
        num_restarts (int): number of restarts to use for the main gmres solver
        tol (float): tolerance for the main gmres solver

    Returns:
        (tuple of scipy.sparse or numpy matrices): solutions from gmres solver
    '''
    local_soln = gmres_solver_wrapper(Ai, fi, gi, Hips,useSchurs, yguess=yguess, niter=niter, 
                                         num_restarts=num_restarts, tol=tol)
    (uli, yli) = local_soln
    combined_local = np.concatenate(local_soln)

    if(iproc == 0):
        left_bound = - len(yli)
        right_bound = F.shape[1]
    else:
        left_bound = (iproc - 1) *len(combined_local)
        right_bound = (iproc) * len(combined_local)

    ul0 = sparse.linalg.spsolve_triangular(B, f.reshape(len(f), 1) - F[:, left_bound:right_bound] * combined_local, False)

    u10 = ul0.reshape(len(ul0), 1)

    return((u10, combined_local))


def gmres_solver_wrapper(Ai, fi, gi, Hips, useSchurs, yguess = None, niter = 10, num_restarts = 10, tol = 1e-6):
    '''
    distributed gmres solver

    Args:
        Ai (scipy.sparse or numpy matrix): Ai variable for gmres per report/website
        fi (scipy.sparse or numpy matrix): fi variable for gmres per report/website
        gi (scipy.sparse or numpy matrix): gi variable for gmres per report/website
        Hips (scipy.sparse or numpy matrix): Hips variable for gmres per report/website
        useSchurs (boolean): boolean to use or not to use schurs in gmres solver
        yguess (scipy.sparse or numpy matrix): the current best guess for y matrix
        niter (int): number of iterations to use for the main gmres solver
        num_restarts (int): number of restarts to use for the main gmres solver
        tol (float): tolerance for the main gmres solver

    Returns:
        (tuple of scipy.sparse or numpy matrices): combined f and g components of gmres result
    '''
    
    nproc = MPI.COMM_WORLD.Get_size()
    iproc = MPI.COMM_WORLD.Get_rank()
    if(len(fi)):
        combined = np.concatenate((fi, gi))
    else:
        combined = gi
    
    thresh = None
    if((Ai.diagonal()).prod() == 0):
        thresh = 0
        
    (L, U, L_inner, U_inner, r) = (None, None, None, None, None)
    
    if(useSchurs):

        r = sparse.linalg.gmres(Ai, combined)[0]
    (Hi, dx, Bi, Hi) = (None, None, None, None)
     
  
    if(yguess == None):
        yguess = np.zeros_like(gi)
        if(r is not None):
            yguess = np.reshape(r, yguess.shape)
    for count in range(0, num_restarts):
        '''Do this communcation part for number of iteration'''
        
        if(nproc == 1):
            adjust_left = np.zeros_like(yguess)
        else:
            interface_y = communicate_interface(iproc, nproc, yguess)
            #Do the dot product
            adjust_left = interface_dotProd(interface_y, Hips)
         
        if(useSchurs):
            Pr = r[len(fi):]
            yguess_new = sparse.linalg.gmres(Ai, combined - adjust_left, Pr, restart = None)[0]
            
        else:
            Hi = Ai[len(fi):, :len(fi)]
            dx = np.linalg.lstsq(Hi.todense(), gi - adjust_left)[0]
            Bi = Ai[:len(fi), :len(fi)]
            HiT = Ai[:len(fi), len(fi):]
            yguess_new = np.linalg.lstsq(HiT.todense(), (fi - Bi * dx))[0]
        
        yguess = np.reshape(yguess_new, yguess.shape)    
        
    
    #Now commmunicate what's left
    if(nproc != 1):
        interface_y = communicate_interface(iproc, nproc, yguess)
        t = interface_dotProd(interface_y, Hips)
    else:
        t = np.zeros_like(gi)
    #Subtract it out and do a final solve
    gi = gi - t
    
    if(not useSchurs):
        Hi = Ai[len(fi):, :len(fi)]
        dx = np.linalg.lstsq(Hi.todense(), gi - adjust_left)[0]
        Bi = Ai[:len(fi), :len(fi)]
        HiT = Ai[:len(fi), len(fi):]
        yguess_new = np.linalg.lstsq(HiT.todense(), (fi - Bi * dx))[0]
        combined_soln = np.concatenate((dx, yguess_new))

    else:        
        if(len(fi)):
            combined = np.concatenate((fi, gi))
        else:
            combined = gi
    
        combined_soln = sparse.linalg.gmres(Ai, combined)[0].reshape(len(combined), 1)
    return(combined_soln[:len(fi)], combined_soln[len(fi):])
    
    
def communicate_interface(iproc, nproc, toSend, useMPI = True):
    '''
    convenience function for communicating interface variables

    Args:
        iproc (int): MPI processor id
        nproc (int): procs for communication interface
        toSend (scipy.sparse matrix or numpy matrice): data to be broadbast
        useMPI (boolean): boolean to determine whether to use MPI when sending data

    Returns:
        (scipy.sparse or numpy matrix): return result from the communicate interface
    '''
    if(not useMPI):
        return([np.zeros_like(toSend)])
    from_other_to_root = toSend
    if(iproc == 0):
        from_other_to_root = None
    cont_to_power_injection = MPI.COMM_WORLD.gather(from_other_to_root, root = 0)
    
    toBcast = None
    if(iproc == 0):
        toBcast = [toSend]
    power_injection_to_cont = MPI.COMM_WORLD.bcast(toBcast, root = 0)
   
    interface_y = None
    if(iproc == 0):
        interface_y = [x for x in cont_to_power_injection if x is not None]
    else:
        interface_y = power_injection_to_cont
    
    return(interface_y)
    
def interface_dotProd(interface_y, Hips):
    '''
    convenience  function for dot produce at the interface
    Args:
        interface_y (scipy.sparse or numpy matrix): interface_y variables per report/website
        Hips (scipy.sparse or numpy matrix): Hips variables per report/website

    Result:
        (float): result of the interface dot product between interface_y and Hips variables
    '''
    adjust_left = 0
    for index in range(0, len(Hips)):
     
        temp = Hips[index] * interface_y[index]
	
        adjust_left += temp
    return(adjust_left)
            

def find_regrouping(model):
    '''
    give the model data, finds a suitable regrouping of the variables

    Args:
        model (scipy.sparse or numpy matrix): model data matrix
    
    Result:
        inds_info_col (list or numpy array of scipy.sparse or numpy matrices):
            column related info about inds
        inds_info_row (list or numpy array of scipy.sparse or numpy matrices):
            row related info about inds
        inds (dict of lists or numpy arrays):
            dict, with key, value pairs where the values are an ordered list of 
            indices defining the regrouping
    '''

    #For the last level after eliminating lambda, now group so that 
    #the thetas/nus are grouped based on contigency number. This is similar 
    #to the other ordering function and should find a way to refactor
    ncont = len(model[0][0][1][0]) - 1
    inds = {'deltaTheta': [], 'deltaNu': [],
            'rDual': [],  'rPri': []}

    shift = 0
    shift2 = 0
    for cons_num in range(0, ncont + 1):
        model_specific = get_model_specific(model, cons_num)
        info_mapping = make_mapping(model_specific)
        dim_mapping = make_mapping(
            model_specific[0][0][info_mapping['dim']][0])
        dims = model_specific[0][0][info_mapping['dim']][0][0]

        dim_th = dims[dim_mapping['th']][0][0]
        dim_neq = dims[dim_mapping['neq']][0][0]

        inds['deltaTheta'].append(np.arange(shift, dim_th + shift))
        shift = shift + dim_th
        inds['rDual'].append(np.arange(shift2, dim_th + shift2))
        shift2 += dim_th

    general_mapping = make_mapping(model[0][0][0][0][0])
    dims_general = model[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)
    p = dims_general[0][0][dims_general_mapping['p']][0][0]
    inds['deltaP'] = np.arange(shift, p+shift)
    shift += p

    for cons_num in range(0, ncont + 1):
        model_specific = get_model_specific(model, cons_num)
        info_mapping = make_mapping(model_specific)
        dim_mapping = make_mapping(
            model_specific[0][0][info_mapping['dim']][0])
        dims = model_specific[0][0][info_mapping['dim']][0][0]

        dim_th = dims[dim_mapping['th']][0][0]
        dim_neq = dims[dim_mapping['neq']][0][0]

        inds['deltaNu'].append(np.arange(shift, dim_neq + shift))
        shift += dim_neq

        inds['rPri'].append(np.arange(shift2, dim_neq + shift2))
        shift2 += dim_neq

    inds['rdualP'] = np.arange(shift2, shift2 + p)
    shift2 += p

    inds_info_col = [np.concatenate((x1, y1)) for x1, y1 in zip(
        inds['deltaTheta'], inds['deltaNu'])]
    inds_info_col.append(inds['deltaP'])
    inds_info_row = [np.concatenate((x1, y1))
                     for x1, y1 in zip(inds['rDual'], inds['rPri'])]
    inds_info_row.append(inds['rdualP'])

    return(inds_info_col, inds_info_row, inds)
