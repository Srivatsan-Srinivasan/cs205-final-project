import os
import logging
#from solvers import solver_multilevel

def run_multilevel_example():

	logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

	logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')

	#run some sample from the dataset
	sample_bus_count = 189
	sample_constr_count = 5

	#solve the system
	os.system('mpirun -n {} python -u multilevel_example_wrapper.py {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count))

	#solver_multilevel.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
	run_multilevel_example()