'''
SERIAL MULTILEVEL SOLVER

Serial Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import sys
import logging
import profile_utils

from solvers import solver_multilevel_serial

def run_serial_multilevel_example():
    '''
    Runs the serial solver (optionally using the newton generation parallelization) given bus count, constraint count and options about parallelization.
    
    Args:
        sample_bus_count (int): number of busses in the power network
        sample_constr_count (int): number of constraints in the power network
        newtonParllel (bool): to parallelize the newton matrix generation step
        newton_nproc (int): number of processors to use in newton generation parallelization. setting to 0 results in full parallelization

    Returns:
        soln (scipy.sparse matrix or numpy matrix): the solution to the system of equations for the newton step
    note: this function is called repeatedly to profile the code
    '''
    #Runs the serial version of the multi level solver given the bus count, constraint count, and options for parallelization of newton generation    
    logging.info('RUNNING SERIAL MULTI-LEVEL SOLVER')

    #solve the system
    return solver_multilevel_serial.solve(sample_bus_count, sample_constr_count, newtonParallel, newton_nproc)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset
    #sample_bus_count = 189
    #sample_constr_count = 5
    #newtonParallel = 'false'
    #newton_nproc = 0

    sample_bus_count = int(sys.argv[1])
    sample_constr_count = int(sys.argv[2])
    newtonParallel = sys.argv[3]
    newton_nproc = int(sys.argv[4])

    newtonParallel = newtonParallel.lower() == 'true'

    #profile function calls
    profile_utils.profile_funcalls(run_serial_multilevel_example)
    #trim the profile of function calls to most interesting groupings
    profile_utils.viz_funcalls()

    #profile the call graph
    profile_utils.profile_callgraph(run_serial_multilevel_example)

    #profile the memory usage of the program
    profile_utils.profile_memory(run_serial_multilevel_example)

    #profile the time taken for the program to execute
    profile_utils.profile_time(run_serial_multilevel_example)
    
    #profile the residuals (a check of the accuracy of the solution) of the calculation
    profile_utils.check_residuals(run_serial_multilevel_example, sample_bus_count, sample_constr_count)