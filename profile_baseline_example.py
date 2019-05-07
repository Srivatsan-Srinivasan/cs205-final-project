'''
SERIAL UNOPTIMIZED SOLVER

Serial Codes for Newton Step Solver in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import sys
import logging
import profile_utils

from solvers import solver_baseline

def run_baseline_example():
        '''
    Runs baseline solver given bus count and constraint count.

    Args:
        sample_bus_count (int): the number of busses in the network
        sample_constr_count (int): the number of constraints in the network

    Returns:
        soln (scipy.sparse matrix or numpy matrix): the solution to the system of equations for the newton step
    note: this function is called repeatedly to profile the code
        
    '''
    #Runs baseline solver given bus count and constraint count.
    logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

    #solve the system
    return solver_baseline.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset

    #sample_bus_count = 189
    #sample_constr_count = 5

    sample_bus_count = int(sys.argv[1])
    sample_constr_count = int(sys.argv[2])

    #profile function calls
    profile_utils.profile_funcalls(run_baseline_example)
    #trim the profile of function calls to most interesting groupings
    profile_utils.viz_funcalls()

    #profile the call graph
    profile_utils.profile_callgraph(run_baseline_example)
    #profile the memory usage of the program
    profile_utils.profile_memory(run_baseline_example)
    #profile the time taken for the program to execute
    profile_utils.profile_time(run_baseline_example)
    #profile the residuals (a check of the accuracy of the solution) of the calculation
    profile_utils.check_residuals(run_baseline_example, sample_bus_count, sample_constr_count)