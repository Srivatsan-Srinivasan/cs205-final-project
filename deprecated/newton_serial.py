'''
Serial Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import os
import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve


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
        ILU = sparse.linalg.spilu(B0 + 1e-4 * sparse.eye(B0.shape[0]))
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
            y0 = repermute(y0, pInput[0][1])
        return y0

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

def run_sample3(model_fname, rhs_fname, y0_fname, constr_fname):
    #touched
    model_data = sio.loadmat(model_fname)
    y0_data = sio.loadmat(y0_fname)
    constr_data = sio.loadmat(constr_fname)
    rhs = sio.loadmat(rhs_fname)['rhs']

    M_n = create_M_newton(model_data, y0_data, constr_data)
    (inds, nrows) = find_reordering(model_data['model'])    
    sTime = time.time()
    mlSolve =  multigrid(M_n, rhs, 0, 1, 'given','given', [inds], nrows )
    eTime = time.time() - sTime
    print('########')
    print('act time for multigrid {}s.'.format(eTime))
    print('########')
    return(mlSolve)


def get_names(b, n, directory):
    end_name = "%d_nbus%d_ncont" %(b, n)
    model_fname = os.path.join(directory, "model_%s" % end_name)
    rhs_fname = os.path.join(directory, "rhs_%s" % end_name)
    y0_fname = os.path.join(directory, "y0_%s" % end_name)
    constr_fname = os.path.join(directory, "constr_%s" % end_name)
    
    return(model_fname, rhs_fname, y0_fname, constr_fname)


def get_res(A, b, x):
    '''Calc the residual'''
    return(np.linalg.norm((A * x).flatten() - b.flatten()))
    
def run_many():
    '''Compute a timing table'''
    #buses = [5, 189, 2224]
    #ncont = [1, 2, 3, 4, 5, 6]

    buses = [189]
    ncont = [5]
    
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


            final_starttime = time.time()
            M_n = create_M_newton(model_data, y0_data, constr_data)
            final_endtime = time.time()
            print('create M newton took {}s for {}, {}'.format(final_endtime - final_starttime, b, n))
            # continue



            # raise TypeError

            start_time = time.time()
            res = run_sample3(model_fname, rhs_fname, y0_fname, constr_fname)            
            end_time = time.time() - start_time

            # #### cprofile logs ####
            # import cProfile
            # cProfile.runctx('run_sample3(model_fname, rhs_fname, y0_fname, constr_fname)', globals(), locals(), '{}.profile'.format(end_name))

            # return None, None
            # #### cprofile logs ####


            # #### pycallgraph logs ####
            # from pycallgraph import PyCallGraph
            # from pycallgraph.output import GraphvizOutput
            # import os

            # with PyCallGraph(output=GraphvizOutput()):
            #     res = run_sample3(model_fname, rhs_fname, y0_fname, constr_fname)            
            # os.rename('pycallgraph.png', 'pycallgraph-{}.png'.format(end_name))
            # #### pycallgraph logs ####

            # #### min, max memory logs ####
            # from memory_profiler import memory_usage

            # def mem_fnwrapper():
            #     return run_sample3(model_fname, rhs_fname, y0_fname, constr_fname)

            # mem_usage = memory_usage(mem_fnwrapper)
            # print(mem_usage)
            # #### min, max memory logs ####

            M_n = create_M_newton(model_data, y0_data, constr_data)

            resid = get_res(M_n, rhs, res)
            timing_dict['Multilevel'][end_name] = end_time
            residual_dict['Multilevel'][end_name] = resid
            

            del model_data
            del y0_data
            del constr_data
            del rhs
            del M_n
            del res
            
            #Direct 
            start_time1 = time.time()

            model_data = sio.loadmat(model_fname)
            y0_data = sio.loadmat(y0_fname)
            constr_data = sio.loadmat(constr_fname)

            rhs = sio.loadmat(rhs_fname)['rhs']

            M_n = create_M_newton(model_data, y0_data, constr_data)
            res = sparse.linalg.lsqr(M_n, rhs)[0]
            end_time1 = time.time() - start_time1            
            resid1 = get_res(M_n, rhs, res)

            print('resid1 match: {}'.format(resid1))
            continue

            del M_n
            del res
            M_n = create_M_newton(model_data, y0_data, constr_data)
            start_timez = time.time()
            res = sparse.linalg.lsqr(M_n, rhs)[0]
            end_timez = time.time() - start_timez


            print('@@@@@@@@')
            print('{},{}'.format(b, n))
            print('normal timing: {}.'.format(end_time1))
            print('normal - multi level timing: {}.  '.format(end_time1 - end_time))
            print('normal / multi level timing: {}.  '.format(end_time1/end_time))
            print('normal - multi level residual: {}.'.format(resid1 - resid))
            print('normal residual: {}.'.format(resid))
            print('@@@@@@@@')
            
            
            print('########')
            print('act time for direct {}s.'.format(end_timez))
            print('########')

            timing_dict['direct'][end_name] = end_time
            residual_dict['direct'][end_name] = resid

            
    #print(np.average(list(residual_dict['Multilevel'].values())))
    #print(np.min(list(residual_dict['Multilevel'].values())))                
    #print(np.max(list(residual_dict['Multilevel'].values())))

    #print([float(i) for i in list(residual_dict['Multilevel'].values())])
    #print(np.average([float(i) for i in list(residual_dict['Multilevel'].values())]))
    #print(np.max([float(i) for i in list(residual_dict['Multilevel'].values())]))


    return(timing_dict, residual_dict)            

#8 = A
#5= B
#9 = D
#1  = dim


def make_mapping(d):
    '''Takes a model like data structure and returns a mapping'''
    key = d.dtype.fields.keys()
    val = np.arange(len(key))
    mapping = {k:v for k, v in zip(key, val)}
    return(mapping)


def get_model_specific(model, cont_num):    
    '''Get the constraint specific model information'''
    return model[0][0][1][0][cont_num]


def create_M_newton(model_data, y0_data, constr_data):    
### touched

    '''Takes in model/y0/cosntr data and constructs
    the newton block diagonal matrix as consistent with the paper.'''

    model = model_data['model']
    y0 = y0_data['y0']

    dim_y0_mapping = make_mapping(y0[0])
    theta = y0[0][0][dim_y0_mapping['th']][0][0]

    Ms = []
    Bu = []
    # Total specific to consider
    total_cons = len(model[0][0][1][0])
    # Create the M and the Bu matri foe each contigency
    for con_num in range(total_cons):
        model_specific = get_model_specific(model, con_num)
        info_mapping = make_mapping(model_specific)
        dim_mapping = make_mapping(
            model_specific[0][0][info_mapping['dim']][0])
        nieq = model_specific[0][0][info_mapping['dim']
                                    ][0][0][dim_mapping['nineq']][0][0]

        # TODO: should use a mapping for A2, D2 etc...
        A2 = model_specific[0][0][8]
        D2 = model_specific[0][0][9]
        B = model_specific[0][0][5]
        m = model_specific[0][0][info_mapping['dim']
                                 ][0][0][dim_mapping['m']][0][0]

        constr_mapping = make_mapping(constr_data['constr'][0])

        gval = np.zeros((nieq, 1))
        gval[:m] = D2.dot(A2).dot(
            theta) - constr_data['constr'][0][0][constr_mapping['uf']][0][con_num]
        gval[m:2*m] = -D2.dot(A2).dot(theta) + \
            constr_data['constr'][0][0][constr_mapping['lf']][0][con_num]
        gval = gval.flatten()

        Dg = np.concatenate((D2.dot(A2), -D2.dot(A2)))
        Ax = B

        dim_th = model_specific[0][0][info_mapping['dim']
                                      ][0][0][dim_mapping['th']][0][0]
        dim_n = model_specific[0][0][info_mapping['dim']
                                     ][0][0][dim_mapping['n']][0][0]

        ab = (y0[0][0][dim_y0_mapping['lam']][0][con_num]).flatten()
        LM = -1 * np.diag(ab)
        LM = LM.dot(Dg)

        r1 = [np.zeros((dim_th, dim_th)),    Dg.T,                       Ax.T]
        r2 = [LM,                 -
              np.diag(gval),               np.zeros((nieq, dim_n))]
        r3 = [Ax,                  np.zeros(
            (dim_n, nieq)),     np.zeros((dim_n, dim_n))]

        Ms.append(sparse.csr_matrix(np.block([r1, r2, r3])))

        Au = -1 * np.eye(dim_n)
        lam_u_s = np.size(y0[0][0][dim_y0_mapping['lam_u']])
        r1 = [np.zeros((dim_th, dim_n)), np.zeros((dim_th, lam_u_s))]
        r2 = [np.zeros((nieq, dim_n)), np.zeros((nieq, lam_u_s))]
        r3 = [Au, np.zeros((dim_n, lam_u_s))]
        Bu.append(sparse.csr_matrix(np.block([r1, r2, r3])))

    # Constructing the general matrix - need to know the variables
    g_u = np.zeros((lam_u_s, 1))

    general_mapping = make_mapping(model[0][0][0][0][0])
    dims_general = model[0][0][0][0][0][general_mapping['dim']]
    dims_general_mapping = make_mapping(dims_general)

    # TODO: Fix naming...
    n = dims_general[0][0][dims_general_mapping['p']][0][0]
    p = y0[0][0][dim_y0_mapping['p']]
    g_u[:n] = p - constr_data['constr'][0][0][constr_mapping['up']]
    g_u[n:2*n] = -p + constr_data['constr'][0][0][constr_mapping['lp']]

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

    # This shift is just keep track of where we need to update in the block matrices
    shift = 0
    for con_num in range(total_cons):
        model_specific_i = get_model_specific(model, con_num)
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

    # Combine it all together!

    '''
        AA: Combination of all the Ms
        Du: Primary/dual variables related to the initial physical construactions
        BB: something in betweenn...(should be a bit more clear)
        '''
    M_newton = sparse.bmat([[AA, sparse.csc_matrix(BB)], [
                           sparse.csc_matrix(BB).T, sparse.csc_matrix(Du)]])
    return(M_newton)

''' Find the reordeirng - note that this is really just grouping the 
 type of variables together. '''


def get_specific_dim(model, cons_num):    
    model_specific = get_model_specific(model, cons_num)
    info_mapping= make_mapping(model_specific)
    dim_mapping = make_mapping(model_specific[0][0][info_mapping['dim']][0])
    dims = model_specific[0][0][info_mapping['dim']][0][0]
    return((dims, dim_mapping))

def find_reordering(model, method = "standard"):
    '''The algorithm here is to permute to get the same type of variables together. 
    Roughtly speaking instead of grouping by contingency directly we want to group by 
    theta variables, lambda variables etc. This somehow gives us a very diagonal 
    B0 matrix which is really easy to work with. 
    '''
    if(method not in ['standard', 'grouped']):
        raise ValueError("Invalid method supplied")
    #Method types are standard (group lambdas together and then group (nu/theta))
    # grouped is grouping all lambdas together than all nus and then all thetas
    ncont = len(model[0][0][1][0]) - 1
    inds = {'deltaTheta': [], 'deltaLambda': [], 'deltaNu': [],
            'rDual': [], 'rCent': [], 'rPri': []}

    shift = 0
    shift2 = 0
    for cons_num in range(0, ncont + 1):
        model_specific = get_model_specific(model, cons_num)
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

    general_mapping = make_mapping(model[0][0][0][0][0])
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

    new_column_inds = np.zeros(shift, dtype='int')
    shift = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_lam = dims[dim_mapping['lam']][0][0]
        new_column_inds[shift:shift + dim_lam] = inds['deltaLambda'][k]
        shift += dim_lam

    new_column_inds[shift: 2 * p + shift] = inds['deltaLambdaP']
    shift += 2 * p
    B0_nrows = shift  # All the lambda changes

    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_th = dims[dim_mapping['th']][0][0]
        new_column_inds[shift:dim_th + shift] = inds['deltaTheta'][k]
        shift += dim_th

        if(method == 'grouped'):
            dim_neq = dims[dim_mapping['neq']][0][0]
            new_column_inds[shift:dim_neq+shift] = inds['deltaNu'][k]
            shift += dim_neq
        
       
    if(method == "standard"):
        for k in range(0, ncont + 1):
            (dims, dim_mapping) = get_specific_dim(model, k)
            dim_neq = dims[dim_mapping['neq']][0][0]          
            new_column_inds[shift:dim_neq+shift] = inds['deltaNu'][k]
            shift += dim_neq
       
    new_column_inds[shift:p+shift] = inds['deltaP']
    shift += p

    new_rows_inds = np.zeros(shift2, dtype='int')

    shift2 = 0
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
        dim_lam = dims[dim_mapping['lam']][0][0]
        new_rows_inds[shift2:dim_lam + shift2] = inds['rCent'][k]
        shift2 += dim_lam

    new_rows_inds[shift2:2*p + shift2] = inds['rcentP']
    shift2 += 2 * p
    for k in range(0, ncont + 1):
        (dims, dim_mapping) = get_specific_dim(model, k)
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
            (dims, dim_mapping) = get_specific_dim(model, k)
            dim_neq= dims[dim_mapping['neq']][0][0]
            new_rows_inds[shift2:dim_neq + shift2] = inds['rPri'][k]
            shift2 += dim_neq

    return([new_rows_inds, new_column_inds, inds], [B0_nrows, dims_general[0][0][dims_general_mapping['th']][0][0]])

if __name__ == '__main__':
    a, b  = run_many()
    print(a)
    print(b)