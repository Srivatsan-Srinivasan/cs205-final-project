import logging
from solvers import solver_baseline

logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

#run some sample from the dataset
sample_bus_count = 189
sample_constr_count = 5

#solve the system
solver_baseline.solve(sample_bus_count, sample_constr_count)