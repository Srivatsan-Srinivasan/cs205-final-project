import sys
import logging
import profile_utils

from solvers import solver_multilevel_serial

def run_serial_multilevel_example():
    logging.info('RUNNING SERIAL MULTI-LEVEL SOLVER')

    #solve the system
    return solver_multilevel_serial.solve(sample_bus_count, sample_constr_count, newtonParallel, newton_nproc)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset
    #sample_bus_count = 189
    sample_bus_count = int(sys.argv[1])
    #sample_constr_count = 5
    sample_constr_count = int(sys.argv[2])

    #newtonParallel = 'false'
    newtonParallel = sys.argv[3]

    newton_nproc = int(sys.argv[4])

    newtonParallel = newtonParallel.lower() == 'true'

    #profile_utils.profile_funcalls(run_serial_multilevel_example)
    #profile_utils.viz_funcalls()

    #profile_utils.profile_callgraph(run_serial_multilevel_example)
    #profile_utils.profile_memory(run_serial_multilevel_example)
    profile_utils.profile_time(run_serial_multilevel_example)
    #profile_utils.check_residuals(run_serial_multilevel_example, sample_bus_count, sample_constr_count)