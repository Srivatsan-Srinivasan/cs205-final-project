import logging
from solvers import solver_multilevel_serial

def run_serial_multilevel_example():

	logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

	logging.info('RUNNING SERIAL MULTI-LEVEL SOLVER')

	#run some sample from the dataset
	sample_bus_count = 189
	sample_constr_count = 5

	#solve the system
	solver_multilevel_serial.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
	run_serial_multilevel_example()