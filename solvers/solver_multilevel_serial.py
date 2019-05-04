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

def _construct_newton(model_data, y0_data, constr_data):
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

    logging.info('finished constructing newton matrix.')    

    return newton_matrix, inds, nrows

def _solver(newton_matrix, inds, nrows, rhs_data):
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

    soln = multigrid(newton_matrix, rhs_data, 0, 1, 'given', 'given', [inds], nrows)

    logging.info('finished solving system.')

    return soln

def solve(bus_count, constr_count):

    # load data, construct newton matrix, solve newton step
    model_data, rhs_data, y0_data, constr_data = _load_data(bus_count, constr_count)

    newton_matrix, inds, nrows = _construct_newton(model_data, y0_data, constr_data)

    return _solver(newton_matrix, inds, nrows, rhs_data)