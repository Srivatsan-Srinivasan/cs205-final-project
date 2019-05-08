'''
GENERAL HELPER FUNCTIONS

Simple Common Utils for Newton Step Solvers in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''
import os

import numpy as np
import scipy.io as sio

def get_filepaths(bus_count, constr_count, directory):
    '''
    Fetches the full form filepaths for data files given number of busses and number 
    of power network constraints.

    Args:
        bus_count (ints): number of busses in power network
        constr_count (ints): number of constraints for power network
        directory (string): base directory hosting data files

    Returns:
        model_fname (string): full form filepath to model data file
        rhs_fname (string): full form filepath to RHS variables (per report/website) variables data file
        y0_fname (string): full form filepath to Y0 variables (per report/website) variables data file
        constr_fname (string): full form filepath to constraint data file
    '''

    end_name = "%d_nbus%d_ncont" %(bus_count, constr_count)

    model_fname = os.path.join(directory, "model_%s" % end_name)
    rhs_fname = os.path.join(directory, "rhs_%s" % end_name)
    y0_fname = os.path.join(directory, "y0_%s" % end_name)
    constr_fname = os.path.join(directory, "constr_%s" % end_name)
    
    return model_fname, rhs_fname, y0_fname, constr_fname

def load_filepaths(model_fname, rhs_fname, y0_fname, constr_fname):
    '''
    Given the full form file paths for data files, loades the MATLAB matrices.

    Args:
        model_fname (string): full form filepath to model data file
        rhs_fname (string): full form filepath to RHS variables (per report/website) variables data file
        y0_fname (string): full form filepath to Y0 variables (per report/website) variables data file
        constr_fname (string): full form filepath to constraint data file

    Returns:
        model_data (numpy matrix): loaded model data file matrix
        rhs_data (numpy matrix): RHS variables (per report/website) variables data file matrix
        y0_data (numpy matrix): Y0 variables (per report/website) variables data file matrix
        constr_data (numpy matrix): constraint data file matrix
    '''
            
    model_data = sio.loadmat(model_fname)['model']
    rhs_data = sio.loadmat(rhs_fname)['rhs']
    y0_data = sio.loadmat(y0_fname)['y0']
    constr_data = sio.loadmat(constr_fname)['constr']

    return model_data, rhs_data, y0_data, constr_data

def calculate_residuals(A, B, x):
    '''
    Calculates the residual for the solution to Ax=B

    Args:
        A (scipy.sparse or numpy matrices): A in Ax=B
        x (scipy.sparse or numpy matrices): x in Ax=B
        B (scipy.sparse or numpy matrices): B in Ax=B

    Returns:
        residual (float): residual for the solution to Ax=B
    '''    
    return(np.linalg.norm((A * x).flatten() - B.flatten()))