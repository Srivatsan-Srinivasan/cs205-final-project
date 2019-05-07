'''
PARALLEL MULTILEVEL SOLVER

Parallel Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import os
import sys
import logging
import profile_utils

def run_multilevel_example():
    '''    
    runs the parallel multilevel newton step solver, helper function for profilign the newton step solver

    Args:
        sample_bus_count (int): the number of busses in the power network
        sample_constr_count (int): the number of constraints in the power network
        fullParallel (bool): parallelize block 3 as well as block 2 in the solver ? (see report/website)
        newtonParallel (bool): parallelize generation of the newton step matrix?
        newton_nproc (int): number of procs to use in parallelization of newton step matrix. 0 defaults to full parallization

    Returns:
        nothing. executes the wrapper for the fully parallelized version of the newton step solver using mpirun        
    '''
    logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')
    

    #solve the system
    os.system('mpirun -n {} python -u multilevel_example_wrapper.py {} {} {} {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count, fullParallel, newtonParallel, newton_nproc))

def multilevel_check_residuals():
    '''    
    runs the parallel multilevel newton step solver, helper function for profilign the residuals of the newton step solver

    Args:
        sample_bus_count (int): the number of busses in the power network
        sample_constr_count (int): the number of constraints in the power network
        fullParallel (bool): parallelize block 3 as well as block 2 in the solver ? (see report/website)
        newtonParallel (bool): parallelize generation of the newton step matrix?
        newton_nproc (int): number of procs to use in parallelization of newton step matrix. 0 defaults to full parallization

    Returns:
        nothing. executes the residual wrapper for the fully parallelized version of the newton step solver using mpirun
    '''
    logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')

    #solve the system
    os.system('mpirun -n {} python -u multilevel_example_residuals_wrapper.py {} {} {} {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count, fullParallel, newtonParallel, newton_nproc))

if __name__ == '__main__':
    '''
    Given the number of constraints in the network, number of busses and various parallelization options, executes the parallel multigrid solver.

    Args:
        sample_bus_count (int): number of busses in the power network
        sample_costr_count (int): number of constraints for the power network
        fullParallel (bool): parallelize block 3 as well as block 2 in the solver ? (see report/website)
        newtonParallel (bool): parallelize generation of the newton step matrix?
        newton_nproc (int): number of procs to use in parallelization of newton step matrix. 0 defaults to full parallization

    Returns:
        Nothing. profiles the code.
    '''
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset

    #sample_bus_count = 2224    
    #sample_constr_count = 6
    #fullParallel = 'false'
    #newtonParallel = 'false'
    #newton_nproc = 0

    sample_bus_count = int(sys.argv[1])
    sample_constr_count = int(sys.argv[2])
    fullParallel = sys.argv[3]
    newtonParallel = sys.argv[4]
    newton_nproc = int(sys.argv[5])

    #profile the time taken for the program to execute
    profile_utils.profile_time(run_multilevel_example)

    #profile the residuals (a check of the accuracy of the solution) of the calculation
    #multilevel_check_residuals()