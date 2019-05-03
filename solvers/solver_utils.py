'''
SOLVER SPECIFIC FUNCTIONS

Solver Specific Utils for Newton Step Solvers in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import numpy as np
from scipy import sparse

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

def construct_NewtonBDMatrix(model_data, y0_data, constr_data):
    '''
    Given the model, some B (in Ax=B) variable, some x (in Ax=B) variables constructs
    the newton block diagonal matrix per paper.

    Args:
        model_data: loaded model data file matrix
        y0_data: some x (in Ax=B) variables data file matrix
        constr_data: constraint data file matrix

    Returns:
        newton_matrix: newton block diagonal matrix per paper
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