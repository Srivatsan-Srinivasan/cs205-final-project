import os
import logging
import profile_utils

#from functools import partial
from solvers import solver_multilevel

#sample_bus_count = None
#sample_constr_count = None

def run_multilevel_example():
    logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')

    #run some sample from the dataset
    #sample_bus_count = 189
    #sample_constr_count = 5

    #solve the system
    os.system('mpirun -n {} python -u multilevel_example_wrapper.py {} {} {} {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count, fullParallel, newtonParallel, newton_nproc))

    #solver_multilevel.solve(sample_bus_count, sample_constr_count)

def multilevel_check_residuals():
    logging.info('RUNNING PARALLEL MULTI-LEVEL SOLVER')

    #run some sample from the dataset
    #sample_bus_count = 189
    #sample_constr_count = 5

    #solve the system
    os.system('mpirun -n {} python -u multilevel_example_residuals_wrapper.py {} {} {} {} {}'.format(sample_constr_count + 2, sample_bus_count, sample_constr_count, fullParallel, newtonParallel, newton_nproc))

    #solver_multilevel.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset
    #global sample_bus_count
    #global sample_constr_count

    sample_bus_count = 2224
    sample_constr_count = 6

    fullParallel = 'false'
    newtonParallel = 'false'
    newton_nproc = 0

    #run_multilevel_example_filled = run_multilevel_example(sample_bus_count, sample_constr_count)
    #multilevel_check_residuals_filled = multilevel_check_residuals(sample_bus_count, sample_constr_count)

    #profile_utils.profile_funcalls(run_multilevel_example)
    #profile_utils.viz_funcalls()

    #profile_utils.profile_callgraph(run_multilevel_example)
    #profile_utils.profile_memory(run_multilevel_example)
    profile_utils.profile_time(run_multilevel_example)
    #profile_utils.check_residuals(run_multilevel_example, sample_bus_count, sample_constr_count)
    #multilevel_check_residuals()