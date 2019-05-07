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

def run_multilevel_example():
    '''
    Given the number of constraints in the network, number of busses and various parallelization options, executes the parallel multigrid solver.

    Args:
        sample_bus_count (int): number of busses in the power network
        sample_costr_count (int): number of constraints for the power network
        fullParallel (bool): parallelize block 3 as well as block 2 in the solver ? (see report/website)
        newtonParallel (bool): parallelize generation of the newton step matrix?
        newton_nproc (int): number of procs to use in parallelization of newton step matrix. 0 defaults to full parallization

    Returns:
        Nothing. simply runs the code.
    '''

	logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

	logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')

	#run some sample from the dataset
	#sample_bus_count = 2224    
    #sample_constr_count = 6
    #fullParallel = 'false'
    #newtonParallel = 'false'
    #newton_nproc = 0

    sample_bus_count = int(sys.argv[1])
    sample_constr_count = int(sys.argv[2])

    sys.argv[3] = sys.argv[3].lower() == 'true'
	sys.argv[4] = sys.argv[4].lower() == 'true'

    fullParallel = sys.argv[3]
    newtonParallel = sys.argv[4]
    newton_nproc = int(sys.argv[5])

	#solve the system	
	os.system('mpirun -n {} python -u multilevel_example_wrapper.py {} {} {} {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count, fullParallel, newtonParallel, newton_nproc))

if __name__ == '__main__':
	run_multilevel_example()