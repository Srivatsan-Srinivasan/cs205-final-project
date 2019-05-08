'''
SERIAL UNOPTIMIZED SOLVER

Serial Codes for Newton Step Solver in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import sys
import logging
from solvers import solver_baseline

def run_baseline_example():	
	'''
	Runs baseline solver given bus count and constraint count.

	Args:
		sample_bus_count (int): the number of busses in the network
		sample_constr_count (int): the number of constraints in the network

	Returns:
		Nothing, just calls the newton step solver.		
	'''
	

	
	logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

	logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

	#run some sample from the dataset
	
	#sample_bus_count = 189
	#sample_constr_count = 5

	sample_bus_count = int(sys.argv[1])
	sample_constr_count = int(sys.argv[2])

	#solve the system
	solver_baseline.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
	run_baseline_example()