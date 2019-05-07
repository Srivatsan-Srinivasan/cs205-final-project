'''
SERIAL MULTILEVEL SOLVER

Serial Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import logging

from .utils import get_filepaths
from .utils import load_filepaths
from .utils import calculate_residuals
from scipy import sparse

from .solver_utils import construct_NewtonBDMatrix
from .solver_utils import permute_NewtonBDMatrix
from .solver_utils import multigrid
from .solver_utils import permute_sparse_matrix

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

def _construct_newton(model_data, y0_data, constr_data):
    '''
    Given the model, Y0 variables (per report/website) variables, constraint data file matrix
    the newton block diagonal matrix per report/website.

    Args:
        model_data (scipy.sparse or numpy matrices): loaded model data file matrix
        y0_data (scipy.sparse or numpy matrices): Y0 variables data file matrix per report/website
        constr_data (scipy.sparse or numpy matrices): constraint data file matrix

    Returns:
        newton_matrix (scipy.sparse or numpy matrices): newton block diagonal matrix per report/website
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates
    '''


    #construct newton block diagonal matrix per report/website
    logging.info('constructing newton matrix ...')

    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)

    inds, nrows = permute_NewtonBDMatrix(model_data, 'standard')    

    logging.info('finished constructing newton matrix.')    

    return newton_matrix, inds, nrows

def _solver(newton_matrix, inds, nrows, rhs_data):
    '''
    Given the newton block diagonal matrix per report/website, RHS variables (per report/website) and
    information about the permuted ordering of the newton matrix, solves the system of equations.

    Args:
        newton_matrix (scipy.sparse or numpy matrix): newton block diagonal matrix (per report/website)
        inds (list of numpy arrays): new ordered indices defining arrangement of permuted matrix
        nrows (list of numpy arrays): new orderings defining shift to track updates
        rhs_data (scipy.sparse or numpy matrix): RHS variables (per report/website)

    Returns:
        soln: Solution to the system of equatoins.
    '''

    #naively find the least-squares solution to the system of equations.
    logging.info('solving system ...')

    soln = multigrid(newton_matrix, rhs_data, 0, 1, 'given', 'given', [inds], nrows)

    logging.info('finished solving system.')

    return soln

def solve(bus_count, constr_count):
    '''
    Given the number of busses and the number of constraints for power network, creates and solve the
    appropriate newton step.

    Args:
        bus_count (int): number of busses in power network
        constr_count (int): number of constraints in power network

    Returns:
        soln (scipy.sparse or numpy matrix): Solution to the system of equations.
    '''

    # load data, construct newton matrix, solve newton step
    model_data, rhs_data, y0_data, constr_data = _load_data(bus_count, constr_count)

    newton_matrix, inds, nrows = _construct_newton(model_data, y0_data, constr_data)

    return _solver(newton_matrix, inds, nrows, rhs_data)