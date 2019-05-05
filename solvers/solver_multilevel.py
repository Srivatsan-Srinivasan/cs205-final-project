'''
SERIAL MULTILEVEL SOLVER

Serial Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''
import logging
from mpi4py import MPI

#from .utils import get_filepaths
from .utils import get_filepaths
from .utils import load_filepaths
from .utils import calculate_residuals
from scipy import sparse

from .solver_utils import construct_NewtonBDMatrix
from .solver_utils import permute_NewtonBDMatrix
from .solver_utils import multigrid_PARALLEL
from .solver_utils import permute_sparse_matrix
from .solver_utils import split_newton_matrix
from .solver_utils import local_schurs

def _load_data(bus_count, constr_count):
    '''
    Given number of busses and number of power network constraints, loades the MATLAB matrices.

    Args:
        bus_count (ints): number of busses in power network
        constr_count (ints): number of constraints for power network

    Returns:
        model_data (numpy matrix): loaded model data file matrix
        rhs_data (numpy matrix): RHS variables (per paper) variables data file matrix
        y0_data (numpy matrix): Y0 variables (per paper) variables data file matrix
        constr_data (numpy matrix): constraint data file matrix
    '''

    #load data
    logging.info('loading data ...')

    fnames = get_filepaths(bus_count, constr_count, "Data")
    model_data, rhs_data, y0_data, constr_data = load_filepaths(*fnames)

    logging.info('finished loading data.')

    return model_data, rhs_data, y0_data, constr_data

def _construct_newton(model_data, rhs_data, y0_data, constr_data):
    '''
    Given the model, Y0 variables (per paper) variables, constraint data file matrix
    the newton block diagonal matrix per paper.

    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        y0_data (scipy.sparse or numpy matrices): Y0 variables data file matrix per paper
        constr_data (scipy.sparse or numpy matrices): constraint data file matrix

    Returns:
        newton_matrix (scipy.sparse or numpy matrices): newton block diagonal matrix per paper
    '''


    #construct newton block diagonal matrix per paper
    logging.info('constructing newton matrix ...')

    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)

    inds, nrows = permute_NewtonBDMatrix(model_data, 'standard')    

    newton_matrix = permute_sparse_matrix(newton_matrix, inds[0], inds[1])    
    rhs_data = rhs_data[inds[0]]
    ncont = len(model_data[0][0][1][0]) - 1

    splits, _ = split_newton_matrix(
            newton_matrix, rhs_data, nrows[0], ncont, None, model_data, inds[2])

    logging.info('finished constructing newton matrix.')    

    return splits, inds, nrows

def _descend_level(iproc, splits):

    logging.info('descending level ...')

    local_data = MPI.COMM_WORLD.scatter(splits, root=0)
    logging.info("proc %d has received some data" % iproc)
    res = local_schurs(local_data)
    combined = MPI.COMM_WORLD.gather(res, root=0)

    logging.info('descended level.')

    return combined

def _solver(iproc, combined, inds, nrows):
    '''
    Given the newton block diagonal matrix per paper, RHS variables (per paper) solves
    the system of equations.

    Args:
        newton_matrix:
        rhs_data:

    Returns:
        soln: Solution to the system of equatoins.
    '''

    #naively find the least-squares solution to the system of equations.
    logging.info('solving system ...')

    soln = multigrid_PARALLEL(iproc, combined, inds, nrows)

    logging.info('finished solving system.')

    return soln

def solve(bus_count, constr_count, save=False):

    # load data, construct newton matrix, solve newton step
    nproc = MPI.COMM_WORLD.Get_size()
    iproc = MPI.COMM_WORLD.Get_rank()
    inode = MPI.Get_processor_name()

    splits = None
    if iproc == 0:
        #master node
        model_data, rhs_data, y0_data, constr_data = _load_data(bus_count, constr_count)
        splits, inds, nrows = _construct_newton(model_data, rhs_data, y0_data, constr_data)

    combined = _descend_level(iproc, splits)

    if iproc == 0:
        soln = _solver(iproc, combined, inds, nrows)
        return soln