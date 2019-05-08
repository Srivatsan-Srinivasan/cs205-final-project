'''
SERIAL UNOPTIMIZED SOLVER

Serial Codes for Newton Step Solver in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import logging

from .utils import get_filepaths
from .utils import load_filepaths
from .utils import calculate_residuals
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .solver_utils import construct_NewtonBDMatrix

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
    '''


    #construct newton block diagonal matrix per report/website
    logging.info('constructing newton matrix ...')

    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)

    logging.info('finished constructing newton matrix.')

    return newton_matrix

def _solver(newton_matrix, rhs_data):
    '''
    Given the newton block diagonal matrix per report/website, RHS variables (per report/website) solves
    the system of equations.

    Args:
        newton_matrix (scipy.sparse or numpy matrix): newton block diagonal matrix (per report/website)
        rhs_data (scipy.sparse or numpy matrix): RHS variables (per report/website)

    Returns:
        soln (scipy.sparse or numpy matrix): Solution to the system of equatoins.
    '''

    #naively find the least-squares solution to the system of equations.
    logging.info('solving system ...')

    soln = sparse.linalg.lsqr(newton_matrix, rhs_data)[0]

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

    newton_matrix = _construct_newton(model_data, y0_data, constr_data)

    return _solver(newton_matrix, rhs_data)