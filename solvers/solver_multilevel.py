'''
PARALLEL MULTILEVEL SOLVER

Parallel Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
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
from .solver_utils import construct_NewtonBDMatrix_PARALLEL
from .solver_utils import permute_NewtonBDMatrix
from .solver_utils import multigrid_PARALLEL
from .solver_utils import permute_sparse_matrix
from .solver_utils import split_newton_matrix
from .solver_utils import local_schurs

from .solver_utils import multigrid_full_parallel

def _load_data(bus_count, constr_count):
    '''
    Given number of busses and number of power network constraints, loades the MATLAB matrices.

    Args:
        bus_count (ints): number of busses in power network
        constr_count (ints): number of constraints for power network

    Returns:
        model_data (numpy matrix): loaded model data file matrix
        rhs_data (numpy matrix): RHS variables (per report/website) variables data file matrix
        y0_data (numpy matrix): Y0 variables (per report/website) variables data file matrix
        constr_data (numpy matrix): constraint data file matrix
    '''

    #load data
    logging.info('loading data ...')

    fnames = get_filepaths(bus_count, constr_count, "Data")
    model_data, rhs_data, y0_data, constr_data = load_filepaths(*fnames)

    logging.info('finished loading data.')

    return model_data, rhs_data, y0_data, constr_data

def _construct_newton(model_data, rhs_data, y0_data, constr_data, newtonParallel, newton_nproc, grouping = 'standard'):
    '''
    Given the model, Y0 variables (per report/website) variables, constraint data file matrix
    the newton block diagonal matrix per report/website.

    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        y0_data (scipy.sparse or numpy matrices): Y0 variables data file matrix per report/website
        constr_data (scipy.sparse or numpy matrices): constraint data file matrix

    Returns:
        splits (list of scipy.sparse or numpy matrices): the constructed and split newton block diagonal matrix for
        parallel execution per report/website
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates
    '''


    #construct newton block diagonal matrix per report/website
    logging.info('constructing newton matrix ...')    
    if newtonParallel:
        logging.info('construct in parallel mode')
        newton_matrix = construct_NewtonBDMatrix_PARALLEL(model_data, y0_data, constr_data, newton_nproc)
    else:
        logging.info('construct in serial mode')
        newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)
    inds, nrows = permute_NewtonBDMatrix(model_data, grouping)    

    newton_matrix = permute_sparse_matrix(newton_matrix, inds[0], inds[1])    
    rhs_data = rhs_data[inds[0]]
    ncont = len(model_data[0][0][1][0]) - 1

    splits, _ = split_newton_matrix(
            newton_matrix, rhs_data, nrows[0], ncont, None, model_data, inds[2])

    logging.info('finished constructing newton matrix.')    

    return splits, inds, nrows

def _descend_level(iproc, splits):
    '''
    Descent step at block 2 per report/website in the parallelization plan.

    Args:
        iproc (int): MPI processor id
        splits (list of scipy.sparse or numpy matrices): the constructed and split newton block diagonal matrix for
        parallel execution per report/website

    Returns:
        combined (list of scipy.sparse numpy matrices): the full result of the local schurs computation at this
        descent step
    '''

    logging.info('descending level ...')

    local_data = MPI.COMM_WORLD.scatter(splits, root=0)
    logging.info("proc %d has received some data" % iproc)
    res = local_schurs(local_data)
    combined = MPI.COMM_WORLD.gather(res, root=0)

    logging.info('descended level.')

    return combined

def _solver(iproc, combined, inds, nrows, model_data, fullParallel = False):
    '''
    Given the newton block diagonal matrix per report/website, RHS variables (per report/website) solves
    the system of equations.

    Args:
        iproc (int): MPI processor id
        combined (list of scipy.sparse numpy matrices): the full result of the local schurs computation at this
        descent step
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates
        model_data (scipy.sparse or numpy matrix): corresponding model data
        fullParallel (bool): run with parallel block 3 (per report/website) or not?

    Returns:
        soln: Solution to the system of equations.
    '''

    #naively find the least-squares solution to the system of equations
    logging.info('solving system ...')
    if fullParallel == False: 
        logging.info("solving bottom level in single node")        
        if iproc == 0:
            soln = multigrid_PARALLEL(iproc, combined, inds, nrows)    
            logging.info('finished solving system.')    
            return soln
        else:
            return None
    else:
        logging.info("solving bottom level in multiple nodes")        
        
        soln = multigrid_full_parallel(iproc, combined, inds, nrows, model_data)    
        if(iproc == 0):
            logging.info('finished solving system.')    
            return soln
        else:
            return None

        
def solve(bus_count, constr_count, fullParallel = False, newtonParallel = False, newton_nproc = 0, grouping = None):
    '''
    Given the number of busses and the number of constraints for power network, and optional arguments
    defining the blocks (see report/website) to parallelize, permutation protocol, etc. creates and solve the
    appropriate newton step.

    Args:
        bus_count (int): number of busses in power network
        constr_count (int): number of constraints in power network
        fullParallel (bool): parallelize block 3?
        newtonParallel (bool): parallelize newton cosntruction step?
        newton_nproc (int): number of procs to use in newton construction parallelization, 0 defaults to
        full parallelism
        grouping (bool): use grouping strategy instead of standard in permutation?

    Returns:        
        soln (scipy.sparse or numpy matrix): solution to the system of equations
    '''


    # load data, construct newton matrix, solve newton step
    nproc = MPI.COMM_WORLD.Get_size()
    iproc = MPI.COMM_WORLD.Get_rank()
    inode = MPI.Get_processor_name()

    splits = None
    inds = None
    nrows = None
    model_data = None
    if(grouping == None):
        grouping = "standard"
        if(fullParallel):
            grouping = "grouped"
    if iproc == 0:
        #master node
        model_data, rhs_data, y0_data, constr_data = _load_data(bus_count, constr_count)
        splits, inds, nrows = _construct_newton(model_data, rhs_data, y0_data, constr_data, newtonParallel = newtonParallel, newton_nproc=newton_nproc, 
                                                 grouping = grouping)

    combined = _descend_level(iproc, splits)

    soln = _solver(iproc, combined, inds, nrows, model_data, fullParallel = fullParallel)
    if iproc == 0:
        return soln
