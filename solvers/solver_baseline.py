'''
SERIAL UNOPTIMIZED SOLVER

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
from scipy.sparse.linalg import lsqr

from .solver_utils import construct_NewtonBDMatrix

def solve(bus_count, constr_count):

    #load data
    logging.info('loading data ...')
    fnames = get_filepaths(bus_count, constr_count, "data")
    model_data, rhs_data, y0_data, constr_data = load_filepaths(*fnames)
    logging.info('finished loading data.')

    #construct newton block diagonal matrix per paper
    logging.info('constructing newton matrix ...')
    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)
    logging.info('finished constructing newton matrix.')

    #naively find the least-squares solution to the system of equations.
    logging.info('solving system ...')
    soln = sparse.linalg.lsqr(newton_matrix, rhs_data)[0]
    logging.info('finished solving system.')

if __name__== '__main__':    
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

    #run some sample from the dataset
    sample_bus_count = 189
    sample_constr_count = 5

    solve(sample_bus_count, sample_constr_count)